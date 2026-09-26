"""trajectory_hash 向后兼容：旧格式轨迹的内容寻址逐字节不变。"""

from adaptive_math.agent.replay import TraceEnvelope, replay, trajectory_hash, verify_hash
from adaptive_math.agent.state import (
    EventKind,
    StepErrorCode,
    TerminationReason,
    TraceEvent,
    Usage,
)
from adaptive_math.agent.trace import Trajectory

# tests/fixtures/golden_traces/direct_correct.json 的锚定常数；该 fixture 的
# content_hash 即由此轨迹算出，任何回归都会同时打破 fixture 回放测试。
GOLDEN_CONTENT_HASH = "851f9da5ac349a14ec1b96619e4d4a52469211f930ff389bd469180c3f95cedb"


def _golden_trajectory() -> Trajectory:
    return Trajectory(
        trace_id="golden-direct",
        task_id="golden-task",
        events=(
            TraceEvent(
                sequence=0,
                kind=EventKind.MODEL_OUTPUT,
                monotonic_ms=0,
                payload={"raw": '<final>{"answer":"2"}</final>', "generated_tokens": 2},
            ),
            TraceEvent(sequence=1, kind=EventKind.FINAL, monotonic_ms=1, payload={"answer": "2"}),
        ),
        final_answer="2",
        termination_reason=TerminationReason.FINAL,
        usage=Usage(steps=1, generated_tokens=2),
        runtime_version="runtime-v1+agent-v1",
    )


def test_legacy_trajectory_hash_is_byte_identical_to_pre_harness_schema() -> None:
    assert trajectory_hash(_golden_trajectory()) == GOLDEN_CONTENT_HASH


def test_harness_spec_hash_changes_content_hash_when_set() -> None:
    legacy = _golden_trajectory()
    harnessed = legacy.model_copy(update={"harness_spec_hash": "b" * 64})

    assert trajectory_hash(harnessed) != trajectory_hash(legacy)


def test_envelope_with_harness_spec_hash_verifies_and_replays() -> None:
    trajectory = _golden_trajectory().model_copy(update={"harness_spec_hash": "b" * 64})
    envelope = TraceEnvelope.create(
        trajectory, prompt_version="agent-v1", created_at="2026-09-18T00:00:00Z"
    )

    assert verify_hash(envelope)
    assert replay(envelope) == trajectory


def test_model_version_changes_content_hash_when_set() -> None:
    legacy = _golden_trajectory()
    versioned = legacy.model_copy(update={"model_version": "ckpt-qwen3-sft-v1"})

    assert trajectory_hash(versioned) != trajectory_hash(legacy)
    assert trajectory_hash(legacy) == GOLDEN_CONTENT_HASH


def test_event_error_code_changes_content_hash_when_set() -> None:
    legacy = _golden_trajectory()
    coded_events = (
        legacy.events[0],
        legacy.events[1].model_copy(update={"error_code": StepErrorCode.ACTION_PARSE_ERROR}),
    )
    coded = legacy.model_copy(update={"events": coded_events})

    assert trajectory_hash(coded) != trajectory_hash(legacy)
    assert trajectory_hash(legacy) == GOLDEN_CONTENT_HASH
