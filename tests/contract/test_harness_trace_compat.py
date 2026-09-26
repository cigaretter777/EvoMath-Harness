"""Harness Evolution Phase H1 契约：旧格式轨迹永续可回放。

契约内容（设计报告 §9）：
- `Trajectory` 增加 `harness_spec_hash` 字段后，字段出现之前录制的轨迹
  （无该键的 JSON）仍可解析、hash 校验与 replay；
- 旧轨迹的 `content_hash` 逐字节不变——本文件锚定 golden fixture 的常数。
"""

from pathlib import Path

from adaptive_math.agent.replay import TraceEnvelope, replay, verify_hash
from adaptive_math.evolution.taxonomy import label_trajectory

GOLDEN_TRACE = Path("tests/fixtures/golden_traces/direct_correct.json")
GOLDEN_CONTENT_HASH = "851f9da5ac349a14ec1b96619e4d4a52469211f930ff389bd469180c3f95cedb"


def test_pre_harness_trace_parses_with_absent_spec_hash() -> None:
    envelope = TraceEnvelope.model_validate_json(GOLDEN_TRACE.read_bytes())

    assert envelope.trajectory.harness_spec_hash is None
    assert envelope.trajectory.runtime_version == "runtime-v1+agent-v1"


def test_pre_harness_trace_content_hash_is_unchanged() -> None:
    envelope = TraceEnvelope.model_validate_json(GOLDEN_TRACE.read_bytes())

    assert envelope.content_hash == GOLDEN_CONTENT_HASH
    assert verify_hash(envelope)


def test_pre_harness_trace_still_replays() -> None:
    envelope = TraceEnvelope.model_validate_json(GOLDEN_TRACE.read_bytes())

    assert replay(envelope).final_answer == "2"


def test_golden_correct_trace_has_no_failure_labels() -> None:
    """Phase H2.5 契约：taxonomy 对 golden 正确轨迹打标为空——失败挖掘
    不会把已知正确的轨迹误判进任何失败类别。"""
    envelope = TraceEnvelope.model_validate_json(GOLDEN_TRACE.read_bytes())

    assert label_trajectory(envelope.trajectory, correct=True) == ()
