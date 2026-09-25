import json
import os
import time
from pathlib import Path

from adaptive_math.verifier.symbolic import compare_expressions
from adaptive_math.verifier.worker import SymbolicWorker, WorkerConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "adversarial_answers.jsonl"


def test_worker_uses_pipe_ipc_without_queue_feeder_threads() -> None:
    source = (Path(__file__).parents[2] / "src" / "adaptive_math" / "verifier" / "worker.py").read_text()

    assert "Pipe(duplex=False)" in source
    assert ".Queue()" not in source


def test_default_worker_budget_supports_the_symbolic_runtime() -> None:
    config = WorkerConfig()

    assert config.memory_mb == 4096
    assert config.timeout_seconds == 8.0
    assert config.cpu_seconds == 8


def _selective_sleeping(prediction: str, reference: str, task_id: str) -> dict[str, object]:
    if task_id.startswith("sleep"):
        time.sleep(10)
    return compare_expressions(prediction, reference, task_id)


def _selective_crashing(prediction: str, reference: str, task_id: str) -> dict[str, object]:
    if task_id.startswith("crash"):
        os._exit(3)
    return compare_expressions(prediction, reference, task_id)


def _selective_burner(prediction: str, reference: str, task_id: str) -> dict[str, object]:
    if task_id.startswith("burn"):
        while True:
            pass  # pragma: no cover
    return compare_expressions(prediction, reference, task_id)


def _immediate_verdict(_prediction: str, _reference: str, _task_id: str) -> dict[str, object]:
    return {
        "status": "correct",
        "reward": 1.0,
        "normalized_prediction": "x",
        "normalized_reference": "x",
        "details": {},
    }


def test_worker_startup_is_not_charged_to_first_request_timeout() -> None:
    worker = SymbolicWorker(
        WorkerConfig(timeout_seconds=0.25, startup_timeout_seconds=30.0),
        comparator=_immediate_verdict,
    )
    try:
        assert worker.compare("x", "x", "first-request")["status"] == "correct"
    finally:
        worker.close()


def test_worker_timeout_kills_and_replaces() -> None:
    worker = SymbolicWorker(WorkerConfig(timeout_seconds=2.0), comparator=_selective_sleeping)
    try:
        # Warm up: startup (spawn + sympy import) is not part of per-request
        # timing, so the assertions below measure only the timeout behavior.
        assert worker.compare("x", "x", "warmup")["status"] == "correct"
        start = time.monotonic()
        verdict = worker.compare("x", "x", "sleep-1")
        elapsed = time.monotonic() - start
        assert verdict["status"] == "timeout"
        assert elapsed < 4.0  # below two times the configured timeout
        # the next request must succeed on a fresh worker
        assert worker.compare("x+x", "2*x", "task-2")["status"] == "correct"
    finally:
        worker.close()


def test_worker_cpu_limit_yields_timeout_without_killing_worker() -> None:
    worker = SymbolicWorker(WorkerConfig(cpu_seconds=0.5), comparator=_selective_burner)
    try:
        # Warm up so the timing assertion excludes spawn and sympy import.
        assert worker.compare("x", "x", "warmup")["status"] == "correct"
        start = time.monotonic()
        verdict = worker.compare("x", "x", "burn-1")
        elapsed = time.monotonic() - start
        assert verdict["status"] == "timeout"
        assert elapsed < 2.0
        # the CPU limit interrupts only the runaway request; the worker survives
        assert worker.compare("x+x", "2*x", "task-2")["status"] == "correct"
    finally:
        worker.close()


def test_worker_crash_is_detected_and_replaced() -> None:
    worker = SymbolicWorker(WorkerConfig(), comparator=_selective_crashing)
    try:
        verdict = worker.compare("x", "x", "crash-1")
        assert verdict["status"] == "internal_error"
        assert worker.compare("x+x", "2*x", "task-2")["status"] == "correct"
    finally:
        worker.close()


def test_adversarial_fixture_rows_are_all_consumed() -> None:
    lines = [line for line in FIXTURE.read_text().splitlines() if line.strip()]
    assert len(lines) > 0
    marker = "/tmp/adaptive_math_adversarial_marker"
    if os.path.exists(marker):
        os.remove(marker)
    worker = SymbolicWorker(WorkerConfig(timeout_seconds=2.0, cpu_seconds=2))
    try:
        for line in lines:
            row = json.loads(line)
            verdict = worker.compare(row["prediction"], row["reference"], "adversarial")
            assert verdict["status"] in row["allowed"], f"{row}: got {verdict['status']}"
            # every row must yield reward 0.0, never a crash or a hang
            assert verdict["reward"] == 0.0
        # no file, process or network side effect may have happened
        assert not os.path.exists(marker)
        # the worker must still be healthy after the adversarial batch
        assert worker.compare("x+x", "2*x", "task-after")["status"] == "correct"
    finally:
        worker.close()
    # fixture consumption must cover every line exactly once
    assert len(lines) == sum(1 for line in FIXTURE.read_text().splitlines() if line.strip())
