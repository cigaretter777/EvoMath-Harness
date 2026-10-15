"""Arm rendering: one message differs, and only in the way the report says."""

import pytest

from adaptive_math.recovery.arms import ARMS, HINT_TEXT, MARKER_TEXT, Arm, render_arm_observation

ORIGINAL = (
    "Tool call failed (execution_error).\n"
    "stdout:\n\nstderr:\nTraceback (most recent call last):\n"
    "ZeroDivisionError: division by zero\n"
)


def test_full_arm_is_the_frozen_text_verbatim() -> None:
    assert render_arm_observation(ORIGINAL, Arm.FULL) == ORIGINAL


def test_prompted_arm_appends_the_fixed_hint() -> None:
    rendered = render_arm_observation(ORIGINAL, Arm.PROMPTED)

    assert rendered == f"{ORIGINAL}\n{HINT_TEXT}"
    assert rendered.startswith(ORIGINAL)
    assert "ZeroDivisionError" in rendered


def test_marker_arm_replaces_the_detail() -> None:
    rendered = render_arm_observation(ORIGINAL, Arm.MARKER)

    assert rendered == MARKER_TEXT
    assert "ZeroDivisionError" not in rendered
    assert "execution_error" not in rendered


def test_the_fixed_texts_match_the_protocol() -> None:
    assert HINT_TEXT == (
        "The tool execution failed. Inspect the error feedback and your previous "
        "code, correct the cause before retrying, and continue solving the "
        "original problem."
    )
    assert MARKER_TEXT == "Tool execution failed."


def test_arms_are_distinct_and_ordered() -> None:
    assert [arm.value for arm in ARMS] == ["F", "P", "M"]


def test_an_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError):
        render_arm_observation(ORIGINAL, "X")  # type: ignore[arg-type]
