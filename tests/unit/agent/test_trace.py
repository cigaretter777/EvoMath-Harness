import pytest
from pydantic import ValidationError

from adaptive_math.agent.state import EventKind, StepErrorCode, TerminationReason, TraceEvent, Usage
from adaptive_math.agent.trace import Trajectory


def _trajectory(**overrides: object) -> Trajectory:
    fields: dict[str, object] = {
        "trace_id": "trace-1",
        "task_id": "unit:task",
        "events": (TraceEvent(sequence=0, kind=EventKind.FINAL, monotonic_ms=5, payload={"answer": "2"}),),
        "final_answer": "2",
        "termination_reason": TerminationReason.FINAL,
        "usage": Usage(steps=1),
        "runtime_version": "runtime-v1",
    }
    fields.update(overrides)
    return Trajectory(**fields)  # type: ignore[arg-type]


def test_trajectory_json_round_trip_preserves_deterministic_event_stream() -> None:
    assert Trajectory.model_validate_json(_trajectory().model_dump_json()) == _trajectory()


def test_harness_spec_hash_defaults_to_none_for_legacy_trajectories() -> None:
    assert _trajectory().harness_spec_hash is None


def test_harness_spec_hash_round_trips_when_set() -> None:
    trajectory = _trajectory(harness_spec_hash="a" * 64)

    assert Trajectory.model_validate_json(trajectory.model_dump_json()) == trajectory


def test_harness_spec_hash_rejects_malformed_values() -> None:
    for bad in ("a" * 63, "A" * 64, "g" * 64, ""):
        with pytest.raises(ValidationError):
            _trajectory(harness_spec_hash=bad)


def test_model_version_defaults_to_none_for_legacy_trajectories() -> None:
    assert _trajectory().model_version is None


def test_model_version_round_trips_when_set() -> None:
    trajectory = _trajectory(model_version="ckpt-qwen3-sft-v1")

    assert Trajectory.model_validate_json(trajectory.model_dump_json()) == trajectory


def test_model_version_rejects_empty_and_overlong_values() -> None:
    for bad in ("", "x" * 129):
        with pytest.raises(ValidationError):
            _trajectory(model_version=bad)


def test_event_error_code_defaults_to_none() -> None:
    event = TraceEvent(sequence=0, kind=EventKind.INVALID_ACTION, monotonic_ms=0)

    assert event.error_code is None


def test_event_error_code_round_trips_when_set() -> None:
    event = TraceEvent(
        sequence=0,
        kind=EventKind.INVALID_ACTION,
        monotonic_ms=0,
        error_code=StepErrorCode.ACTION_PARSE_ERROR,
    )

    assert TraceEvent.model_validate_json(event.model_dump_json()) == event


def test_event_error_code_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(
            {
                "sequence": 0,
                "kind": "invalid_action",
                "monotonic_ms": 0,
                "payload": {},
                "error_code": "not_a_real_code",
            }
        )
