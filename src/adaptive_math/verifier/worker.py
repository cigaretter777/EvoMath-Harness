"""Isolated worker process for symbolic verification.

Untrusted expressions are parsed and compared in a spawned worker process.
Three independent limits protect the parent:

- a bounded startup handshake keeps dependency import out of per-request timing;
- a per-request CPU timer (SIGPROF) inside the ready worker interrupts a runaway
  comparison and returns a timeout verdict while the worker survives;
- a parent-side wall-clock deadline kills and replaces the whole worker if
  it stops responding (SIGPROF-blocked C calls, hangs) or crashes.

Resource limits are applied best-effort; RLIMIT_AS is not enforced on macOS,
where the wall-clock and CPU limits still guarantee bounded work.
"""

import multiprocessing as mp
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

# Symbolic verification imports NumPy/SciPy transitively.  Constrain their thread
# pools before those imports run: each verifier is already isolated in a process,
# and allowing every worker to create a full CPU-sized pool exhausts constrained
# cloud containers long before the verifier's own timeout is reached.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from adaptive_math.verifier.symbolic import _parse_to_sympy, compare_expressions

Comparator = Callable[[str, str, str], dict[str, object]]


@dataclass(frozen=True)
class WorkerConfig:
    # A relation-vs-value comparison measures 2.5-2.6s CPU on this host
    # (campaign-20260925 probe; ~3.9s end-to-end with worker spawn), and the
    # wall clock inflates under load, so 2.0/2 tripped the deadline on an
    # answer whose logic was already correct. 8.0/8 keeps ~3x headroom while
    # the CPU timer still bounds a runaway comparison.
    timeout_seconds: float = 8.0
    startup_timeout_seconds: float = 30.0
    # SciPy/SymPy's virtual-memory footprint exceeds 512 MiB during normal
    # expression parsing.  Keep an explicit ceiling, but give an isolated
    # verifier enough room to complete real OpenR1 examples.
    memory_mb: int = 4096
    cpu_seconds: int = 8


class _CpuLimitExceeded(Exception):
    pass


def _with_cpu_limit(
    cpu_seconds: int, comparator: Comparator, prediction: str, reference: str, task_id: str
) -> dict[str, object]:
    def handler(signum: int, frame: object) -> None:
        raise _CpuLimitExceeded

    previous = signal.signal(signal.SIGPROF, handler)
    signal.setitimer(signal.ITIMER_PROF, cpu_seconds, 0)
    try:
        return comparator(prediction, reference, task_id)
    except _CpuLimitExceeded:
        return _error_verdict(
            "timeout", f"cpu limit {cpu_seconds}s exceeded inside the worker"
        )
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, previous)


def _worker_main(
    task_receiver: Any,
    result_sender: Any,
    ready_sender: Any,
    comparator: Comparator,
    config: WorkerConfig,
) -> None:
    _apply_limits(config)
    # Prime the LaTeX parser before signaling ready: its lazy initialization
    # otherwise lands inside the first request's CPU budget and can trip the
    # limit under load. Startup is bounded separately by the handshake deadline
    # and is not charged to per-request timing.
    try:
        _parse_to_sympy(r"x^2 + 1")
    except Exception:  # pragma: no cover - warm-up must never block readiness
        pass
    ready_sender.send(True)
    while True:
        item = task_receiver.recv()
        if item is None:  # shutdown sentinel
            break
        request_id, prediction, reference, task_id = item
        try:
            verdict = _with_cpu_limit(
                config.cpu_seconds, comparator, prediction, reference, task_id
            )
        except BaseException as exc:  # the worker must never die on bad input
            verdict = _error_verdict(
                "internal_error", f"comparator raised {type(exc).__name__}"
            )
        try:
            result_sender.send((request_id, verdict))
        except Exception:
            break  # parent is gone


def _apply_limits(config: WorkerConfig) -> None:
    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS,
            (config.memory_mb * 1024 * 1024, config.memory_mb * 1024 * 1024),
        )
    except (ImportError, ValueError, OSError):
        pass  # macOS does not enforce RLIMIT_AS; timeouts still bound the work


def _error_verdict(status: str, reason: str) -> dict[str, object]:
    return {
        "status": status,
        "reward": 0.0,
        "normalized_prediction": None,
        "normalized_reference": None,
        "details": {"reason": reason},
    }


class SymbolicWorker:
    """Owns one spawned worker process; replaces it after timeout or crash."""

    def __init__(self, config: WorkerConfig | None = None, comparator: Comparator | None = None) -> None:
        self._config = config or WorkerConfig()
        self._comparator = comparator or compare_expressions
        self._ctx = mp.get_context("spawn")
        self._task_sender: Any = None
        self._result_receiver: Any = None
        self._ready_receiver: Any = None
        self._process: Any = None
        self._next_id = 0

    def compare(self, prediction: str, reference: str, task_id: str) -> dict[str, object]:
        if not self._ensure_worker():
            return _error_verdict("internal_error", "worker failed to become ready")
        request_id = self._next_id
        self._next_id += 1
        self._task_sender.send((request_id, prediction, reference, task_id))
        deadline = time.monotonic() + self._config.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                verdict = _error_verdict("timeout", "worker wall-clock deadline exceeded")
                self._restart()
                return verdict
            if not self._result_receiver.poll(min(0.1, remaining)):
                if not self._process.is_alive():
                    verdict = _error_verdict("internal_error", "worker process died")
                    self._reset()
                    return verdict
                continue
            try:
                got_id, payload = self._result_receiver.recv()
            except EOFError:
                verdict = _error_verdict("internal_error", "worker process died")
                self._reset()
                return verdict
            if got_id != request_id:
                continue  # stale result from a previous generation
            return cast("dict[str, object]", payload)

    def close(self) -> None:
        self._shutdown()

    def _ensure_worker(self) -> bool:
        if self._process is not None and self._process.is_alive():
            return True
        if self._process is not None:
            self._join_quietly(self._process)
        task_receiver, self._task_sender = self._ctx.Pipe(duplex=False)
        self._result_receiver, result_sender = self._ctx.Pipe(duplex=False)
        self._ready_receiver, ready_sender = self._ctx.Pipe(duplex=False)
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(
                task_receiver,
                result_sender,
                ready_sender,
                self._comparator,
                self._config,
            ),
            daemon=True,
        )
        self._process.start()
        deadline = time.monotonic() + self._config.startup_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._restart()
                return False
            if not self._ready_receiver.poll(min(0.1, remaining)):
                if not self._process.is_alive():
                    self._reset()
                    return False
                continue
            try:
                self._ready_receiver.recv()
            except EOFError:
                self._reset()
                return False
            return bool(self._process.is_alive())

    def _restart(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._join_quietly(self._process)
            if self._process.is_alive():
                self._process.kill()
                self._join_quietly(self._process)
        self._process = None

    def _reset(self) -> None:
        if self._process is not None:
            self._join_quietly(self._process)
        self._process = None

    def _shutdown(self) -> None:
        if self._process is None:
            return
        try:
            self._task_sender.send(None)
        except Exception:
            pass
        self._process.terminate()
        self._join_quietly(self._process)
        self._process = None

    @staticmethod
    def _join_quietly(process: Any) -> None:
        try:
            process.join(timeout=5)
        except Exception:
            pass
