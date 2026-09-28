"""Tests for the tolerance re-scoring harness.

Why these exist: the harness's value depends on two properties that are easy to
get subtly wrong. The coercion rung must read the digits the model wrote rather
than a re-rendered float, and it must not fire on shapes whose defect is not a
type (a string answer, a boolean, a second final tag, trailing prose) -- a rung
that quietly accepts more than it claims would inflate every arm it touches. And
the strict rung must refuse to score at all when it cannot reproduce the
recorded statuses, otherwise the rungs above it are measured against a harness
that is not the one that produced the file.

The agent genre adds one more way to go quietly wrong: a trajectory stores both
an answer and the turns that led to it, so reading the wrong one -- mining every
turn instead of the last, or re-extracting instead of using what the loop
recorded -- would score a run nobody performed.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from adaptive_math.core.types import AnswerType, ReferenceAnswer

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "analysis" / "tolerance_rescore.py"

REFERENCE = ReferenceAnswer(value="71", answer_type=AnswerType.INTEGER)


def load_script():
    # No dataclasses here, so the repo's usual loader pattern works as-is.
    spec = spec_from_file_location("tolerance_rescore", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rescore():
    return load_script()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('<final>{"answer":71}</final>', "71"),
        ('<final>{"answer": 71}', "71"),
        ('<think>...</think>\n<final>{"answer":-3.5}</final>', "-3.5"),
        # The literal is preserved, not re-rendered through a float: str(1.50)
        # would say 1.5 and str(1e3) would say 1000.0, inventing text.
        ('<final>{"answer":1.50}</final>', "1.50"),
        ('<final>{"answer":1e3}</final>', "1e3"),
    ],
)
def test_coercion_reads_the_literal_digits(rescore, raw, expected):
    assert rescore.numeric_answer_candidate(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        '<final>{"answer":"71"}</final>',  # already the schema's own shape
        '<final>{"answer":true}</final>',  # a boolean is not a written number
        '<final>{"result":71}</final>',  # no answer field
        '<final>{"answer":71}</final>\n<final>{"answer":71}</final>',  # ambiguous
        '<final>{"answer":71} trailing prose</final>',  # body is not one object
        "the answer is 71",  # no tag at all
        '<tool_call>{"answer":71}</tool_call>',  # wrong tag
    ],
)
def test_coercion_declines_shapes_it_does_not_claim(rescore, raw):
    assert rescore.numeric_answer_candidate(raw) is None


def _row(task_id: str, raw: str, extract_status: str, verifier_status: str) -> dict:
    """One row in the canonical read shape, as ``arm_rows`` produces it."""
    return {
        "task_id": task_id,
        "text": raw,
        "recorded_extract_status": extract_status,
        "recorded_status": verifier_status,
        "strict_answer": None,
    }


def _stored_row(task_id: str, raw: str, extract_status: str, verifier_status: str) -> dict:
    """One direct-arm record as the eval writer leaves it on disk."""
    return {
        "task_id": task_id,
        "raw_output": raw,
        "extract_status": extract_status,
        "verifier_status": verifier_status,
    }


def test_the_rungs_nest_and_only_fire_where_the_one_below_missed(rescore):
    rows = [
        # strict extracts, so nothing above it may claim this task
        _row("t1", '<final>{"answer": "71"}</final>', "ok", "correct"),
        # the defect this harness exists for: a closed block with a numeric answer
        _row("t2", '<final>{"answer":71}</final>', "missing", "invalid_prediction"),
        # no marker at all: only the interpreting rung reaches this one
        _row("t3", "So the answer is 71.", "missing", "invalid_prediction"),
        # nothing to extract at any rung
        _row("t4", "I am still thinking about it.", "missing", "invalid_prediction"),
    ]
    scored = rescore.score_arm(
        "synthetic", rows, {"t1": REFERENCE, "t2": REFERENCE, "t3": REFERENCE, "t4": REFERENCE}
    )
    assert scored["protocols"]["strict"]["correct"] == 1
    assert scored["protocols"]["coerce"]["correct"] == 2
    assert scored["protocols"]["lenient"]["correct"] == 3
    assert scored["protocols"]["lenient"]["gained_vs_strict"] == 2
    assert scored["n_coerced_answers"] == 1


def test_a_row_the_harness_cannot_reproduce_is_fatal(rescore):
    # The record says an answer was extracted; this harness extracts nothing.
    rows = [_row("t1", "no marker here at all", "ok", "correct")]
    with pytest.raises(RuntimeError, match="not the one that scored the file"):
        rescore.score_arm("lying", rows, {"t1": REFERENCE})


def test_a_recorded_verdict_the_harness_cannot_reproduce_is_fatal(rescore):
    # Extraction agrees, but the recorded verdict crosses the correct boundary:
    # a headline accuracy would depend on which verifier version scored it.
    rows = [_row("t1", '<final>{"answer": "70"}</final>', "ok", "correct")]
    with pytest.raises(RuntimeError, match="not the one that scored the file"):
        rescore.score_arm("lying", rows, {"t1": REFERENCE})

    # The other direction of the same boundary: the harness would have scored a
    # task correct that the record did not.
    rows = [_row("t1", '<final>{"answer": "71"}</final>', "ok", "invalid_prediction")]
    with pytest.raises(RuntimeError, match="not the one that scored the file"):
        rescore.score_arm("lying", rows, {"t1": REFERENCE})


def test_drift_between_two_non_correct_statuses_is_measured_not_fatal(rescore):
    # The recorded r0/r2 verdicts come from an older verifier: the same rows now
    # come back incorrect where the record says invalid_prediction. Neither is
    # correct, so no accuracy moves -- it is reported, not reconciled away.
    rows = [
        _row("t1", '<final>{"answer": "70"}</final>', "ok", "invalid_prediction"),
        _row("t2", "nothing to extract", "missing", "invalid_prediction"),
    ]
    scored = rescore.score_arm("drifted", rows, {"t1": REFERENCE, "t2": REFERENCE})
    assert scored["protocols"]["strict"]["correct"] == 0
    assert scored["verifier_drift"] == ["t1: incorrect != invalid_prediction"]


def _trajectory(task_id: str, turns: list[str], final_answer: str | None, status: str) -> dict:
    """One stored trajectory, as the rollout writer leaves it on disk."""
    return {
        "task_id": task_id,
        "verdict": {"status": status},
        "trajectory": {
            "final_answer": final_answer,
            "events": [{"kind": "model_output", "payload": {"raw": t}} for t in turns],
        },
    }


def _write(tmp_path: Path, name: str, records: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_arm_rows_reads_a_trajectory_by_its_last_turn(rescore, tmp_path):
    path = _write(
        tmp_path,
        "trajectories.jsonl",
        [
            _trajectory(
                "t1", ["first turn", 'last turn <final>{"answer": "71"}</final>'], "71", "correct"
            )
        ],
    )
    mode, rows = rescore.arm_rows(path)
    assert mode == "agent"
    assert rows[0]["strict_answer"] == "71"
    assert rows[0]["recorded_status"] == "correct"
    # The rungs above strict read the model's final word, not its first thought.
    assert rows[0]["text"] == 'last turn <final>{"answer": "71"}</final>'


def test_arm_rows_reads_a_prediction_file(rescore, tmp_path):
    path = _write(
        tmp_path, "p.jsonl", [_stored_row("t1", "hello", "missing", "invalid_prediction")]
    )
    mode, rows = rescore.arm_rows(path)
    assert mode == "direct"
    assert rows[0]["recorded_extract_status"] == "missing"
    assert rows[0]["strict_answer"] is None


def test_an_agent_row_without_a_recorded_verdict_is_fatal(rescore, tmp_path):
    record = _trajectory("t1", ["a turn"], "71", "correct")
    record["verdict"] = None
    path = _write(tmp_path, "trajectories.jsonl", [record])
    with pytest.raises(RuntimeError, match="carries no verdict"):
        rescore.arm_rows(path)


def test_a_trajectory_with_no_model_turns_is_fatal(rescore, tmp_path):
    path = _write(tmp_path, "trajectories.jsonl", [_trajectory("t1", [], "71", "correct")])
    with pytest.raises(RuntimeError, match="has no model turns"):
        rescore.arm_rows(path)


def test_agent_strict_rung_is_the_recorded_answer_not_a_re_extraction(rescore, tmp_path):
    # The loop closed with "71", so the strict rung must count that -- not go
    # back to the text and derive its own answer, which would score a harness
    # the run did not use.
    record = _trajectory(
        "t1", ['<think>reasoning</think><final>{"answer": "71"}</final>'], "71", "correct"
    )
    _, rows = rescore.arm_rows(_write(tmp_path, "trajectories.jsonl", [record]))
    scored = rescore.score_arm("agent", rows, {"t1": REFERENCE})
    assert scored["mode"] == "agent"
    assert scored["protocols"]["strict"]["correct"] == 1


def test_an_agent_record_the_parser_does_not_reproduce_is_fatal(rescore, tmp_path):
    # The trajectory closed with 71, but the strict parser reads nothing off
    # that last turn -- the shape a run launched with a tolerant parser leaves.
    # Scoring it here would measure a ruler the run did not use, so: stop.
    record = _trajectory(
        "t1", ['<think>reasoning never closed<final>{"answer": "71"}</final>'], "71", "correct"
    )
    _, rows = rescore.arm_rows(_write(tmp_path, "trajectories.jsonl", [record]))
    with pytest.raises(RuntimeError, match="not the one that scored the file"):
        rescore.score_arm("tolerant", rows, {"t1": REFERENCE})


def test_agent_coercion_reads_a_numeric_answer_the_parser_refused(rescore, tmp_path):
    # The shape the agent arms produced most: a well-formed final block whose
    # answer is a JSON number, behind a reasoning block with no closing tag.
    # The strict parser refuses the turn, so the loop recorded no answer; the
    # coercion rung reads the digits off the same text.
    record = _trajectory(
        "t1",
        ['<think>reasoning never closed\n<final>{"answer":71}</final>'],
        None,
        "invalid_prediction",
    )
    _, rows = rescore.arm_rows(_write(tmp_path, "trajectories.jsonl", [record]))
    scored = rescore.score_arm("agent", rows, {"t1": REFERENCE})
    assert scored["protocols"]["strict"]["correct"] == 0
    assert scored["protocols"]["coerce"]["correct"] == 1
    assert scored["n_coerced_answers"] == 1


def test_agent_rungs_do_not_mine_earlier_turns(rescore, tmp_path):
    # Turn one contains the right answer inside a repairable block; the last
    # turn abandons the attempt. Scanning every turn would score a run the
    # model did not produce, so all three rungs must find nothing here.
    earlier = '<let me try <final>{"answer":71}</final>'
    assert rescore.numeric_answer_candidate(earlier) == "71"  # repairable, if read
    record = _trajectory(
        "t1", [earlier, "I cannot determine the value."], None, "invalid_prediction"
    )
    mode, rows = rescore.arm_rows(_write(tmp_path, "trajectories.jsonl", [record]))
    assert mode == "agent"
    scored = rescore.score_arm("agent", rows, {"t1": REFERENCE})
    assert [scored["protocols"][p]["correct"] for p in ("strict", "coerce", "lenient")] == [0, 0, 0]


def test_a_file_mixing_genres_is_fatal(rescore):
    rows = [
        _row("t1", '<final>{"answer": "71"}</final>', "ok", "correct"),
        {
            "task_id": "t2",
            "recorded_status": "correct",
            "recorded_extract_status": None,
            "strict_answer": "71",
            "text": "whatever",
        },
    ]
    with pytest.raises(RuntimeError, match="mixes direct generations and trajectories"):
        rescore.score_arm("mixed", rows, {"t1": REFERENCE, "t2": REFERENCE})
