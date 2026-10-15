"""Stable seed derivation for the feedback-recovery diagnostic.

Everything that needs randomness in this study derives it from the frozen main
seed through :func:`derive_seed`, so a rerun of the same stage with the same
inputs replays the same schedule. ``random.Random`` and ``hashlib`` are used
because both are process-independent: ``hash()`` is randomized per interpreter
(PYTHONHASHSEED) and would silently break reproducibility across restarts.
"""

import hashlib

MAIN_SEED = 20261014

# 2**31 - 1 keeps every derived seed inside the signed 32-bit range that the
# torch / numpy generators accept without a warning or a cast.
_SEED_MODULUS = 2**31 - 1

_PURPOSES = ("task_selection", "state_selection", "run_order", "generation")


def derive_seed(purpose: str, *parts: str) -> int:
    """Map a purpose plus its key parts to a stable seed in [0, 2**31-1).

    The key parts are joined with a byte that cannot occur in a task id or a
    sample id, so ("a", "bc") and ("ab", "c") cannot collide."""
    if purpose not in _PURPOSES:
        raise ValueError(f"unknown seed purpose {purpose!r}; known: {_PURPOSES}")
    payload = "\x1f".join((f"seed-v1:{MAIN_SEED}", purpose, *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % _SEED_MODULUS


def generation_seed(
    *, stage: str, task_id: str, sample_id: str, arm: str, turn: int
) -> int:
    """Per-generation seed. The key carries the whole identity of a logical
    sample (stage, task, sample, arm, turn) so no two generations share a
    stream, and no arm inherits another arm's randomness."""
    if turn < 0:
        raise ValueError("turn must be non-negative")
    return derive_seed("generation", stage, task_id, sample_id, arm, str(turn))


def stable_order_key(purpose: str, *parts: str) -> str:
    """Hex ordering key for deterministic, outcome-blind selection (used for
    picking one anchor per task and for rotating the run order)."""
    if purpose not in ("task_selection", "state_selection", "run_order"):
        raise ValueError(f"purpose {purpose!r} has no stable ordering")
    payload = "\x1f".join((f"seed-v1:{MAIN_SEED}", purpose, *parts)).encode()
    return hashlib.sha256(payload).hexdigest()
