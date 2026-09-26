"""Bounded parser for one optional think block and one top-level action."""

import hashlib
import re
from collections.abc import Sequence
from enum import StrEnum

import orjson
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from adaptive_math.agent.actions import AgentAction
from adaptive_math.core.types import JSONValue

MAX_RAW_CHARS = 32768
MAX_REASONING_CHARS = 24576
_ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)
_ENVELOPE = re.compile(
    r"(?:<think>(?P<think>(?:(?!<think>|</think>).)*)</think>)?"
    r"\s*"
    r"(?:<tool_call>(?P<tool>.*?)</tool_call>|<final>(?P<final>.*?)</final>)",
    re.DOTALL,
)

# Opt-in tolerance rules (campaign 2026-09-24). Each rewrites one malformed but
# unambiguous shape before the existing checks; the fullmatch still decides.
TOLERANCE_RULES: tuple[str, ...] = ("unclosed_think", "bare_final_scalar")


class ParseErrorCode(StrEnum):
    EMPTY = "empty"
    TOO_LONG = "too_long"
    INVALID_ENVELOPE = "invalid_envelope"
    MULTIPLE_ACTIONS = "multiple_actions"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"


class ParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str | None
    action: AgentAction | None
    error: ParseErrorCode | None
    raw_hash: str
    details: dict[str, JSONValue] = Field(default_factory=dict)


def parse_action(raw: str, *, tolerate: Sequence[str] = ()) -> ParseResult:
    """Parse a protocol turn without interpreting or executing its payload.

    Tolerance is opt-in and only widens recovery, never the envelope contract:
    the rules rewrite the text before matching, and the fullmatch still decides.
    The default stays byte-identical to the strict parser.
    """
    raw_hash = hashlib.sha256(raw.encode()).hexdigest()
    text = raw.strip()
    if not text:
        return _failure(raw_hash, ParseErrorCode.EMPTY)
    if len(text) > MAX_RAW_CHARS:
        return _failure(raw_hash, ParseErrorCode.TOO_LONG)
    unknown = sorted(set(tolerate) - set(TOLERANCE_RULES))
    if unknown:
        raise ValueError(f"unknown tolerate rule(s) {unknown}; valid: {TOLERANCE_RULES}")
    action_count = text.count("<tool_call>") + text.count("<final>")
    if action_count > 1:
        return _failure(raw_hash, ParseErrorCode.MULTIPLE_ACTIONS)
    applied: list[str] = []
    if "unclosed_think" in tolerate:
        closed = _close_unclosed_think(text)
        if closed is not None:
            text = closed
            applied.append("unclosed_think")
    match = _ENVELOPE.fullmatch(text)
    if match is None:
        return _failure(raw_hash, ParseErrorCode.INVALID_ENVELOPE)
    reasoning = match.group("think")
    if applied and reasoning is not None:
        reasoning = reasoning.rstrip()
    if reasoning is not None and len(reasoning) > MAX_REASONING_CHARS:
        return _failure(raw_hash, ParseErrorCode.INVALID_SCHEMA)
    payload_text = match.group("tool") or match.group("final")
    try:
        payload = orjson.loads(payload_text)
    except orjson.JSONDecodeError:
        return _failure(raw_hash, ParseErrorCode.INVALID_JSON)
    if not isinstance(payload, dict):
        if "bare_final_scalar" in tolerate and match.group("final") is not None:
            applied.append("bare_final_scalar")
            payload = {"answer": orjson.dumps(payload).decode("utf-8")}
        else:
            return _failure(raw_hash, ParseErrorCode.INVALID_SCHEMA)
    candidate: dict[str, object]
    if match.group("tool") is not None:
        candidate = {"kind": "tool_call", "call": payload}
    else:
        candidate = {"kind": "final", **payload}
    try:
        action = _ACTION_ADAPTER.validate_python(candidate)
    except ValueError:
        return _failure(raw_hash, ParseErrorCode.INVALID_SCHEMA)
    return ParseResult(
        reasoning=reasoning,
        action=action,
        error=None,
        raw_hash=raw_hash,
        details={"tolerance_applied": applied} if applied else {},
    )


def _close_unclosed_think(text: str) -> str | None:
    """Close a lone unclosed <think> right before the first action tag.

    Conservative: only when the body up to the action tag contains no further
    '<', so the closing point is unambiguous. Returns None when the tolerance
    does not apply (rule never fires; the strict path decides).
    """
    if text.count("<think>") != 1 or "</think>" in text:
        return None
    body_start = len("<think>")
    for tag in ("<tool_call>", "<final>"):
        index = text.find(tag, body_start)
        if index != -1 and "<" not in text[body_start:index]:
            return text[:index] + "</think>" + text[index:]
    return None


def _failure(raw_hash: str, error: ParseErrorCode) -> ParseResult:
    return ParseResult(reasoning=None, action=None, error=error, raw_hash=raw_hash)
