"""Measure the gap between tool-call intent and execution in recorded agent runs.

Why this exists
---------------
The 2026-09-26 ``base+tool`` arm records 7 executed tool calls over 200 tasks
(``tool_calls_total = 7``). Read literally, that says the model does not use
tools, and it was read that way: it is the reason the ``sft_dp_v2`` plan tried
to inject sympy demonstrations into the SFT data. Re-parsing the recorded raw
turns says something different. Hundreds of turns carry a complete,
schema-valid ``<tool_call>`` payload that never executed, and the largest
single cause is a missing closing tag. The two readings imply opposite next
steps ("teach it to use tools" vs. "repair the envelope it already uses"), so
the distinction is worth a script rather than an impression.

Everything here is a re-parse of bytes a stored arm already produced. Nothing
is re-executed and nothing is regenerated. The repair levels (auto-closing a
tag, extracting the first block out of surrounding prose) are **counterfactual
upper bounds over recorded text**: under a repair the trajectory would diverge,
so these numbers bound what a protocol repair could unlock, and they are not a
claim about what the arm would have scored.

Self-check, and why the gate is hard
------------------------------------
``executed`` computed here must equal the arm's recorded ``usage.tool_calls``
on every trajectory. If it does not, this script's parser is not the parser
that ran the arm, and the funnel describes a different protocol than the one
the run obeyed. That is a fatal condition, not a warning: the whole point is
that these counts can be trusted against the stored summary.

Usage:
    python scripts/analysis/protocol_gap.py \
        --arm base_tool=artifacts/rollout_health/thesis_e0_base_tool/trajectories.jsonl \
        --arm rule=artifacts/rollout_health/thesis_e0_rule_strategy/trajectories.jsonl \
        --out artifacts/results/mvp-20260929/protocol-gap.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from adaptive_math.agent.actions import FinalAction, ToolAction, ToolCall
from adaptive_math.agent.parser import parse_action

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
FINAL_OPEN = "<final>"
FINAL_CLOSE = "</final>"

# Classes for turns that contain an action tag. A turn's class is decided by
# its FIRST action tag, so the classes are disjoint and the funnel counts add
# up: every turn is exactly one of these, or ``no_action``, or ``executed``.
ACTION_CLASSES = (
    "executed",  # strict parse produced an action; this is what the run counted
    "closed_payload_valid",  # well-formed block, valid payload, still not executed
    "closed_payload_invalid",  # well-formed block, payload fails the schema
    "closed_payload_not_json",  # well-formed block, body is not JSON at all
    "closed_trailing_content",  # well-formed block, more actions follow in the same turn
    "unclosed_payload_valid",  # auto-closing the tag alone makes it executable
    "unclosed_payload_invalid",  # payload decodes and fails the schema
    "unclosed_trailing_content",  # payload decodes, more actions follow
    "unclosed_truncated",  # body starts as JSON and does not finish
    "mentioned_in_prose",  # the tag is discussed, no payload follows
)
NO_ACTION = "no_action"

# Repair levels, tracked per tag: a level is a set of turns that a mechanical
# repair would make executable. Tag-scoped because the two funnels answer
# different questions -- "how many tool calls did the envelope eat" is not
# "how many final answers did it eat" -- and a merged count would silently add
# a handful of finals to a tool number.
#
# The three levels form a ladder: each set contains the one before it, so a
# turn counted at ``auto_close`` is also counted at ``extract_first_block``.
# That is the point -- they are escalating ceilings, and the difference between
# two rungs is the worth of the repair the lower rung lacks.
LEVEL_AUTO_CLOSE = "auto_close"  # close the missing tag, change nothing else
LEVEL_EXTRACT_FIRST = "extract_first_block"  # + take the first block, ignore prose/stacking
LEVEL_COERCE_TYPES = "coerce_types"  # + read a scalar answer literal as its string form


def level_key(tag: str, level: str) -> str:
    """Key for a repair level, e.g. ``tool:auto_close``."""
    return f"{tag}:{level}"


def analyze_turn(raw: str) -> tuple[str, str | None, frozenset[str]]:
    """Classify one turn into ``(class, executed_kind, repair_levels)``.

    One parse decides everything, so the class, the executed count and the
    repair levels can never disagree about what a turn is.
    """
    action = parse_action(raw).action
    levels = _repair_levels(raw)
    if isinstance(action, ToolAction):
        return "executed", "tool", levels
    if isinstance(action, FinalAction):
        return "executed", "final", levels
    tool_at = raw.find(TOOL_OPEN)
    final_at = raw.find(FINAL_OPEN)
    if tool_at == -1 and final_at == -1:
        return NO_ACTION, None, levels
    if tool_at != -1 and (final_at == -1 or tool_at < final_at):
        return "tool_" + _classify_block(raw, TOOL_OPEN, TOOL_CLOSE, tag="tool"), None, levels
    return "final_" + _classify_block(raw, FINAL_OPEN, FINAL_CLOSE, tag="final"), None, levels


def classify_turn(raw: str) -> str:
    """Classify one model turn. See ACTION_CLASSES for the vocabulary."""
    return analyze_turn(raw)[0]


def _classify_block(raw: str, open_tag: str, close_tag: str, *, tag: str) -> str:
    start = raw.find(open_tag)
    body = raw[start + len(open_tag) :]
    closed = close_tag in body
    if closed:
        body = body.split(close_tag, 1)[0]
    decoded = _decode(body)
    if decoded is None:
        if closed:
            return "closed_payload_not_json"
        # A body that starts as JSON and does not finish was cut off; anything
        # else is prose that merely names the tag.
        return "unclosed_truncated" if body.lstrip().startswith("{") else "mentioned_in_prose"
    value, rest = decoded
    if rest:
        return "closed_trailing_content" if closed else "unclosed_trailing_content"
    valid = _valid_payload(tag, value)
    if closed:
        return "closed_payload_valid" if valid else "closed_payload_invalid"
    return "unclosed_payload_valid" if valid else "unclosed_payload_invalid"


def _decode(body: str) -> tuple[Any, str] | None:
    """Decode the JSON value at the start of ``body``.

    Returns ``(value, rest)`` or None when the body does not start with a
    complete JSON value. ``rest`` is the text after the value, which is empty
    for a clean block and non-empty when the model stacked another action into
    the same turn.
    """
    stripped = body.lstrip()
    if not stripped.startswith("{"):
        return None
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    return value, stripped[end:].strip()


def _valid_payload(tag: str, value: Any) -> bool:
    """Validate a block payload against the schema its tag implies."""
    if not isinstance(value, dict):
        return False
    try:
        if tag == "tool":
            ToolCall.model_validate(value)
        else:
            FinalAction.model_validate({"kind": "final", **value})
    except (ValueError, TypeError):
        return False
    return True


def _coerceable_answer(value: Any) -> bool:
    """Is this payload one scalar-answer type defect away from schema-valid?

    Narrow on purpose: only a JSON number where ``FinalAction.answer`` wants a
    string. Rewriting a number as its own literal text adds no information and
    no interpretation, which is what makes it a repair rather than a guess.
    """
    if not isinstance(value, dict):
        return False
    answer = value.get("answer")
    return isinstance(answer, (int, float)) and not isinstance(answer, bool)


def _repair_levels(text: str) -> frozenset[str]:
    """Which mechanical repairs would make this turn executable, if any.

    ``extract_first_block`` applies whenever the first block carries a
    schema-valid payload, including turns that already executed -- it is a
    ceiling, and ``executed`` is a subset of it. ``auto_close`` is the strictly
    smaller repair: the block is complete and clean, only the closing tag is
    missing. ``coerce_types`` sits above ``extract_first_block`` and adds the
    answer-type repair, so the three sets nest.
    """
    levels: set[str] = set()
    for open_tag, close_tag, tag in (
        (TOOL_OPEN, TOOL_CLOSE, "tool"),
        (FINAL_OPEN, FINAL_CLOSE, "final"),
    ):
        start = text.find(open_tag)
        if start == -1:
            continue
        body = text[start + len(open_tag) :]
        closed = close_tag in body
        if closed:
            body = body.split(close_tag, 1)[0]
        decoded = _decode(body)
        if decoded is None:
            continue
        value, rest = decoded
        valid = _valid_payload(tag, value)
        coerceable = not valid and _coerceable_answer(value)
        if valid:
            levels.add(level_key(tag, LEVEL_EXTRACT_FIRST))
        if valid or coerceable:
            levels.add(level_key(tag, LEVEL_COERCE_TYPES))
        if valid and not closed and not rest:
            levels.add(level_key(tag, LEVEL_AUTO_CLOSE))
    return frozenset(levels)


def analyze_arm(path: Path) -> dict[str, Any]:
    """Re-parse one arm's trajectories into the intent-to-execution funnel."""
    classes: Counter[str] = Counter()
    repairs: Counter[str] = Counter()
    tasks_total = 0
    tasks_with_attempt = 0
    tasks_with_executable_attempt = 0
    turns_total = 0
    executed_tool_calls = 0
    executed_final_actions = 0
    recorded_tool_calls = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tasks_total += 1
        usage = record["trajectory"]["usage"]
        recorded_tool_calls += int(usage["tool_calls"])
        attempt = False
        executable = False
        for event in record["trajectory"]["events"]:
            if event["kind"] != "model_output":
                continue
            turns_total += 1
            raw = event["payload"]["raw"]
            label, executed_kind, levels = analyze_turn(raw)
            classes[label] += 1
            for level in levels:
                repairs[level] += 1
            if executed_kind == "tool":
                executed_tool_calls += 1
                attempt = True
            elif executed_kind == "final":
                executed_final_actions += 1
            elif label.startswith("tool_") and label != "tool_mentioned_in_prose":
                attempt = True
            if attempt and level_key("tool", LEVEL_EXTRACT_FIRST) in levels:
                executable = True
        tasks_with_attempt += int(attempt)
        tasks_with_executable_attempt += int(executable)

    if executed_tool_calls != recorded_tool_calls:
        raise RuntimeError(
            f"{path}: re-parsed {executed_tool_calls} executed tool calls but the arm "
            f"recorded {recorded_tool_calls}; this script is not using the parser that ran it"
        )

    # "Attempt" counts every turn that tried a tool action, executed or not.
    tool_attempt_turns = executed_tool_calls + sum(
        count
        for label, count in classes.items()
        if label.startswith("tool_") and label != "tool_mentioned_in_prose"
    )
    return {
        "trajectories": tasks_total,
        "turns": turns_total,
        "recorded_tool_calls": recorded_tool_calls,
        "executed_tool_calls": executed_tool_calls,
        "executed_final_actions": executed_final_actions,
        "tool_attempt_turns": tool_attempt_turns,
        "tasks_with_tool_attempt": tasks_with_attempt,
        "tasks_with_executable_tool_attempt": tasks_with_executable_attempt,
        "classes": dict(sorted(classes.items())),
        "repair_levels": dict(sorted(repairs.items())),
    }


def _format_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        (
            "| arm | turns | executed tool | executed final | tool attempts | final auto-close | "
            "final extract-first | final + coerce | tool extract-first | tasks w/ executable attempt |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, data in results.items():
        levels = data["repair_levels"]
        lines.append(
            f"| {arm} | {data['turns']} | {data['executed_tool_calls']} | "
            f"{data['executed_final_actions']} | {data['tool_attempt_turns']} | "
            f"{levels.get(level_key('final', LEVEL_AUTO_CLOSE), 0)} | "
            f"{levels.get(level_key('final', LEVEL_EXTRACT_FIRST), 0)} | "
            f"{levels.get(level_key('final', LEVEL_COERCE_TYPES), 0)} | "
            f"{levels.get(level_key('tool', LEVEL_EXTRACT_FIRST), 0)} | "
            f"{data['tasks_with_executable_tool_attempt']}/{data['trajectories']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=TRAJECTORIES_JSONL",
        help="repeatable; NAME labels the arm in the output",
    )
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    args = parser.parse_args(argv)

    arms: list[tuple[str, Path]] = []
    for spec in args.arm:
        name, _, raw_path = spec.partition("=")
        if not name or not raw_path:
            print(f"usage error: --arm expects NAME=PATH, got {spec!r}")
            return 2
        arms.append((name, Path(raw_path)))

    results: dict[str, dict[str, Any]] = {}
    for name, path in arms:
        results[name] = analyze_arm(path)

    print(_format_table(results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arms": results}, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
