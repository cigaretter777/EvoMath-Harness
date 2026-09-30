"""Tests for the protocol-gap funnel analysis.

Why these exist: the funnel's whole value is that its counts can be held
against a stored arm's recorded summary, and that its classes partition the
turns. Both properties are easy to break silently -- a classifier that
mis-files a shape does not crash, it just moves hundreds of turns between
buckets and changes the story. Each malformed shape the model actually
produced in the 2026-09-26 arms is pinned here, plus the arithmetic that the
classes add up, plus the fatal self-check on executed counts.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "analysis" / "protocol_gap.py"

EXECUTED_TOOL = '<tool_call>{"name": "sympy", "arguments": {"operation": "solve"}}</tool_call>'
EXECUTED_FINAL = '<final>{"answer": "12"}</final>'


def load_script():
    spec = spec_from_file_location("protocol_gap", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gap():
    return load_script()


def test_executed_turn_is_classified_by_its_own_parse(gap):
    assert gap.classify_turn(EXECUTED_TOOL) == "executed"
    assert gap.classify_turn(EXECUTED_FINAL) == "executed"
    assert gap.analyze_turn(EXECUTED_TOOL)[1] == "tool"
    assert gap.analyze_turn(EXECUTED_FINAL)[1] == "final"


def test_unclosed_payload_is_recoverable_by_auto_close_alone(gap):
    # The shape the base+tool arm produced most: complete payload, no closing
    # tag, generation stops right after the JSON.
    raw = '<think>\n</think>\n\n<tool_call>{"name":"sympy","arguments":{"operation":"solve"}}'
    label, kind, levels = gap.analyze_turn(raw)
    assert label == "tool_unclosed_payload_valid"
    assert kind is None
    assert levels == frozenset(
        {
            gap.level_key("tool", gap.LEVEL_AUTO_CLOSE),
            gap.level_key("tool", gap.LEVEL_EXTRACT_FIRST),
            # The rungs nest, so the top one is in the set by construction: a
            # turn the bottom rung already recovers needs no coercion.
            gap.level_key("tool", gap.LEVEL_COERCE_TYPES),
        }
    )


def test_stacked_calls_need_extraction_not_just_a_closing_tag(gap):
    raw = (
        '<tool_call>{"name":"sympy","arguments":{"operation":"solve"}}\n\n'
        '<tool_call>{"name":"sympy","arguments":{"operation":"simplify"}}'
    )
    label, _, levels = gap.analyze_turn(raw)
    assert label == "tool_unclosed_trailing_content"
    assert gap.level_key("tool", gap.LEVEL_EXTRACT_FIRST) in levels
    assert gap.level_key("tool", gap.LEVEL_AUTO_CLOSE) not in levels


def test_a_scalar_answer_is_one_coercion_away_from_executable(gap):
    # The shape that dominated the final-action failures: a well-formed, closed
    # block whose only defect is that the answer is a JSON number where
    # FinalAction wants a string.
    raw = '<think>\nThe answer is 71.\n</think>\n\n<final>{"answer":71}</final>'
    label, kind, levels = gap.analyze_turn(raw)
    assert label == "final_closed_payload_invalid"
    assert kind is None
    assert gap.level_key("final", gap.LEVEL_COERCE_TYPES) in levels
    # Coercion is a rung above extraction: closing a tag that is already closed
    # and pulling out a block that is already alone repair nothing.
    assert gap.level_key("final", gap.LEVEL_EXTRACT_FIRST) not in levels
    assert gap.level_key("final", gap.LEVEL_AUTO_CLOSE) not in levels


def test_repair_levels_nest(gap):
    # A clean unclosed valid block sits at the bottom rung, so it must also be
    # counted by both rungs above it: they are ceilings, not categories.
    raw = '<final>{"answer": "7"}'
    levels = gap.analyze_turn(raw)[2]
    assert {
        gap.level_key("final", gap.LEVEL_AUTO_CLOSE),
        gap.level_key("final", gap.LEVEL_EXTRACT_FIRST),
        gap.level_key("final", gap.LEVEL_COERCE_TYPES),
    } <= levels


def test_coercion_does_not_rescue_other_payload_defects(gap):
    # No answer field at all, and a non-scalar answer: neither is a type defect,
    # so neither may be counted as one coercion away from valid.
    for raw in ('<final>{"result": 71}</final>', '<final>{"answer": [71]}</final>'):
        assert gap.analyze_turn(raw)[2] == frozenset()
    # A boolean is not a number being written out; str(True) would invent text.
    assert gap.analyze_turn('<final>{"answer": true}</final>')[2] == frozenset()


def test_truncated_payload_recovers_nothing(gap):
    raw = '<tool_call>{"name":"sympy","argum'
    label, _, levels = gap.analyze_turn(raw)
    assert label == "tool_unclosed_truncated"
    assert levels == frozenset()


def test_prose_mention_is_not_an_attempt(gap):
    raw = "The <tool_call> tag must be closed, so I will not use it here."
    label, _, levels = gap.analyze_turn(raw)
    assert label == "tool_mentioned_in_prose"
    assert levels == frozenset()


def test_closed_block_rejected_by_the_envelope_is_still_an_attempt(gap):
    # Strict parsing is a fullmatch: prose around a well-formed block kills it,
    # even though the block itself is executable.
    raw = 'Let me compute this.\n<tool_call>{"name": "sympy", "arguments": {}}</tool_call>'
    label, kind, levels = gap.analyze_turn(raw)
    assert label == "tool_closed_payload_valid"
    assert kind is None
    assert gap.level_key("tool", gap.LEVEL_EXTRACT_FIRST) in levels


def test_payload_schema_is_checked_against_the_tag(gap):
    # Uppercase tool name fails ToolCall's pattern; a final block must carry
    # "answer" and is not allowed to satisfy the tool schema instead.
    assert gap.classify_turn('<tool_call>{"name": "SymPy"}</tool_call>') == (
        "tool_closed_payload_invalid"
    )
    assert gap.classify_turn('<final>{"name": "sympy", "arguments": {}}</final>') == (
        "final_closed_payload_invalid"
    )
    assert gap.classify_turn("<tool_call>solve x</tool_call>") == "tool_closed_payload_not_json"
    assert gap.classify_turn('<final>{"answer":"7"}') == "final_unclosed_payload_valid"


def test_plain_prose_is_no_action(gap):
    assert gap.classify_turn("Just some reasoning, no action at all.") == gap.NO_ACTION


@given(st.text(max_size=500))
def test_classification_never_raises_on_arbitrary_text(gap, raw):
    label, _, _ = gap.analyze_turn(raw)
    assert label == gap.NO_ACTION or label == "executed" or "_" in label


def _write_arm(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "trajectories.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _record(task_id: str, raws: list[str], recorded_tool_calls: int) -> dict:
    return {
        "task_id": task_id,
        "trajectory": {
            "usage": {"tool_calls": recorded_tool_calls},
            "events": [{"kind": "model_output", "payload": {"raw": raw}} for raw in raws],
        },
    }


def test_funnel_counts_and_task_spread(gap, tmp_path):
    arm = _write_arm(
        tmp_path,
        [
            _record("t1", [EXECUTED_TOOL, EXECUTED_FINAL], recorded_tool_calls=1),
            _record(
                "t2",
                [
                    '<tool_call>{"name":"sympy","arguments":{}}',
                    "plain prose with no action",
                ],
                recorded_tool_calls=0,
            ),
        ],
    )
    data = gap.analyze_arm(arm)
    assert data["turns"] == 4
    assert data["executed_tool_calls"] == 1
    assert data["executed_final_actions"] == 1
    assert data["tool_attempt_turns"] == 2
    assert data["tasks_with_tool_attempt"] == 2
    assert data["tasks_with_executable_tool_attempt"] == 2
    assert data["classes"][gap.NO_ACTION] == 1
    assert data["classes"]["tool_unclosed_payload_valid"] == 1


def test_executed_count_must_match_the_recorded_usage(gap, tmp_path):
    # The arm recorded a tool call that this parser cannot reproduce: the
    # funnel would describe a different protocol, so this must be fatal.
    arm = _write_arm(tmp_path, [_record("t1", [EXECUTED_FINAL], recorded_tool_calls=1)])
    with pytest.raises(RuntimeError, match="not using the parser that ran it"):
        gap.analyze_arm(arm)
