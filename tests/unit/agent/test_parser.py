import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from adaptive_math.agent.actions import FinalAction, ToolAction
from adaptive_math.agent.parser import ParseErrorCode, parse_action


@pytest.mark.parametrize(
    ("raw", "action_type", "reasoning"),
    [
        ('<final>{"answer":"42"}</final>', FinalAction, None),
        (
            '<think>Use symbolic simplification.</think><tool_call>{"name":"sympy","arguments":{"expression":"x**2 - 1"}}</tool_call>',
            ToolAction,
            "Use symbolic simplification.",
        ),
        (
            '<tool_call>{"name":"python","arguments":{"code":"print(1 + 2)\\nprint(3)"}}</tool_call>',
            ToolAction,
            None,
        ),
    ],
)
def test_parse_action_accepts_exactly_one_valid_action(
    raw: str, action_type: type[FinalAction] | type[ToolAction], reasoning: str | None
) -> None:
    result = parse_action(raw)

    assert isinstance(result.action, action_type)
    assert result.reasoning == reasoning
    assert result.error is None
    assert result.raw_hash == hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("", ParseErrorCode.EMPTY),
        ("explain <final>{\"answer\":\"1\"}</final>", ParseErrorCode.INVALID_ENVELOPE),
        ("<think>x</think><think>y</think><final>{\"answer\":\"1\"}</final>", ParseErrorCode.INVALID_ENVELOPE),
        ("<final>{\"answer\":\"1\"}</final><think>x</think>", ParseErrorCode.INVALID_ENVELOPE),
        ("<final>{\"answer\":\"1\"}</final><final>{\"answer\":\"2\"}</final>", ParseErrorCode.MULTIPLE_ACTIONS),
        ("<tool_call>{\"name\":\"python\",\"arguments\":[]}</tool_call>", ParseErrorCode.INVALID_SCHEMA),
        ("<final>{not json}</final>", ParseErrorCode.INVALID_JSON),
        ("<unknown>{}</unknown>", ParseErrorCode.INVALID_ENVELOPE),
        ("<final>{\"answer\":\"1\"}</final> trailing", ParseErrorCode.INVALID_ENVELOPE),
        ("<final>{\"answer\":\"1\"}", ParseErrorCode.INVALID_ENVELOPE),
        ("x" * 32769, ParseErrorCode.TOO_LONG),
    ],
)
def test_parse_action_rejects_invalid_envelopes(raw: str, error: ParseErrorCode) -> None:
    result = parse_action(raw)

    assert result.action is None
    assert result.error == error


@settings(max_examples=100)
@given(st.text(max_size=40000))
def test_parse_action_never_raises_for_arbitrary_text(raw: str) -> None:
    result = parse_action(raw)

    assert result.raw_hash == hashlib.sha256(raw.encode()).hexdigest()


def test_parser_allows_whitespace_between_think_and_action() -> None:
    result = parse_action(
        '<think>2+2=4</think>\n\n<final>{"answer":"4"}</final>'
    )

    assert result.error is None
    assert result.action is not None
    assert result.reasoning == "2+2=4"


# Protocol tolerance as an opt-in harness field (campaign 2026-09-24).
#
# On the R2 rollout-health diagnostic 21 of 72 model turns opened a think block
# and never closed it, and 6 more put a bare scalar inside the final tag. Both
# shapes carry one unambiguous action; the strict fullmatch rejects them, and one
# trajectory re-emitted the identical rejected string until its step budget ran
# out. Tolerance defaults to off so the champion path stays byte-identical.


TC_OPEN = '<tool_call>'
TC_CLOSE = '</tool_call>'

UNCLOSED_THINK_FINAL = '<think>' + '\n\n' + '<final>{"answer":"130"}' + '</final>'
UNCLOSED_THINK_TOOL = (
    '<think>' + 'b - 168 = 0 so b = 168\n'
    + TC_OPEN
    + '{"name":"sympy","arguments":{"operation":"solve","expression":"b-168"}}'
    + TC_CLOSE
)
BARE_FINAL_SCALAR = "<final>3024</final>"
TWO_FINALS = '<final>{"answer":"1"}</final><final>{"answer":"2"}</final>'
PROSE_AROUND_UNCLOSED_THINK = (
    'explain ' + '<think>' + '\n\n<final>{"answer":"1"}' + '</final> trailing prose'
)
ALL_TOLERANCES = ("unclosed_think", "bare_final_scalar")


def test_parse_action_stays_strict_by_default() -> None:
    """The champion path must not change: tolerance is opt-in."""
    assert parse_action(UNCLOSED_THINK_FINAL).error is ParseErrorCode.INVALID_ENVELOPE
    assert parse_action(BARE_FINAL_SCALAR).error is ParseErrorCode.INVALID_SCHEMA


def test_unclosed_think_tolerance_recovers_the_final_action() -> None:
    result = parse_action(UNCLOSED_THINK_FINAL, tolerate=("unclosed_think",))

    assert result.error is None
    assert isinstance(result.action, FinalAction)
    assert result.action.answer == "130"


def test_unclosed_think_tolerance_recovers_a_tool_call_and_keeps_reasoning() -> None:
    result = parse_action(UNCLOSED_THINK_TOOL, tolerate=("unclosed_think",))

    assert result.error is None
    assert isinstance(result.action, ToolAction)
    assert result.action.call.name == "sympy"
    assert result.reasoning == "b - 168 = 0 so b = 168"


def test_bare_final_scalar_tolerance_recovers_the_answer() -> None:
    result = parse_action(BARE_FINAL_SCALAR, tolerate=("bare_final_scalar",))

    assert result.error is None
    assert isinstance(result.action, FinalAction)
    assert result.action.answer == "3024"


def test_tolerance_records_which_rule_fired() -> None:
    result = parse_action(UNCLOSED_THINK_FINAL, tolerate=("unclosed_think",))

    assert result.details["tolerance_applied"] == ["unclosed_think"]
    assert parse_action('<final>{"answer":"1"}</final>').details == {}


def test_tolerance_never_accepts_two_actions() -> None:
    result = parse_action(TWO_FINALS, tolerate=ALL_TOLERANCES)

    assert result.error is ParseErrorCode.MULTIPLE_ACTIONS


def test_tolerance_never_accepts_prose_outside_the_envelope() -> None:
    result = parse_action(PROSE_AROUND_UNCLOSED_THINK, tolerate=ALL_TOLERANCES)

    assert result.error is ParseErrorCode.INVALID_ENVELOPE


def test_unknown_tolerance_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="tolerate"):
        parse_action('<final>{"answer":"1"}</final>', tolerate=("nope",))


@settings(max_examples=100)
@given(st.text(max_size=40000))
def test_tolerance_never_raises_for_arbitrary_text(raw: str) -> None:
    result = parse_action(raw, tolerate=ALL_TOLERANCES)

    assert result.raw_hash == hashlib.sha256(raw.encode()).hexdigest()


def test_both_tolerances_can_fire_in_one_turn() -> None:
    result = parse_action("<think><final>3024</final>", tolerate=ALL_TOLERANCES)

    assert result.error is None
    assert isinstance(result.action, FinalAction)
    assert result.action.answer == "3024"
    assert result.details["tolerance_applied"] == ["unclosed_think", "bare_final_scalar"]


def test_unclosed_think_without_action_stays_rejected() -> None:
    result = parse_action("<think>just thinking", tolerate=("unclosed_think",))

    assert result.error is ParseErrorCode.INVALID_ENVELOPE
