"""The three continuation arms (report section 7).

The intervention is a single edit of the anchor's tool observation. Everything
before it is byte-identical across arms, and every later tool result reverts to
the frozen protocol, so an arm difference can only come from this one message.
"""

from enum import StrEnum

HINT_TEXT = (
    "The tool execution failed. Inspect the error feedback and your previous "
    "code, correct the cause before retrying, and continue solving the "
    "original problem."
)

MARKER_TEXT = "Tool execution failed."


class Arm(StrEnum):
    """F = full original feedback, P = F plus the fixed hint, M = bare marker."""

    FULL = "F"
    PROMPTED = "P"
    MARKER = "M"


ARMS: tuple[Arm, ...] = (Arm.FULL, Arm.PROMPTED, Arm.MARKER)


def render_arm_observation(original_observation: str, arm: Arm) -> str:
    """The anchor observation as this arm's model sees it.

    F keeps the frozen rendering untouched (it is what stage B's own model saw
    at the same state, so the natural continuation and arm F coincide). P
    appends the fixed hint after a newline. M replaces the whole content with
    the fixed marker, including the ``Tool call failed (...)`` line: the marker
    is the arm's only error signal, which is what the report's fixed-marker arm
    describes -- and, as section 1 states, F-M therefore mixes error detail
    with length rather than isolating one of them.
    """
    if arm is Arm.FULL:
        return original_observation
    if arm is Arm.PROMPTED:
        return f"{original_observation}\n{HINT_TEXT}"
    if arm is Arm.MARKER:
        return MARKER_TEXT
    raise ValueError(f"unknown arm {arm!r}")


def is_faithful_full_arm(original: str, rendered: str) -> bool:
    """Guard for the F arm's contract: it must be the frozen text verbatim."""
    return original == rendered
