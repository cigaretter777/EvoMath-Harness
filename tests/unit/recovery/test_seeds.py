"""Seed derivation: stable across processes, isolated per sample and turn."""

import subprocess
import sys

from adaptive_math.recovery.seeds import MAIN_SEED, derive_seed, generation_seed, stable_order_key


def test_the_main_seed_is_the_protocol_value() -> None:
    assert MAIN_SEED == 20261014


def test_derived_seeds_are_stable_across_processes() -> None:
    """``hash()`` would change with PYTHONHASHSEED; the derivation must not."""
    script = (
        "from adaptive_math.recovery.seeds import generation_seed;"
        "print(generation_seed(stage='C', task_id='t', sample_id='s', arm='F', turn=3))"
    )
    seeds = set()
    for hashseed in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
        )
        seeds.add(completed.stdout.strip())

    assert len(seeds) == 1
    assert seeds == {str(generation_seed(stage="C", task_id="t", sample_id="s", arm="F", turn=3))}


def test_every_part_of_the_generation_key_changes_the_seed() -> None:
    base = generation_seed(stage="C", task_id="t", sample_id="s", arm="F", turn=0)

    assert generation_seed(stage="B", task_id="t", sample_id="s", arm="F", turn=0) != base
    assert generation_seed(stage="C", task_id="u", sample_id="s", arm="F", turn=0) != base
    assert generation_seed(stage="C", task_id="t", sample_id="u", arm="F", turn=0) != base
    assert generation_seed(stage="C", task_id="t", sample_id="s", arm="P", turn=0) != base
    assert generation_seed(stage="C", task_id="t", sample_id="s", arm="F", turn=1) != base


def test_key_parts_cannot_be_reassembled_into_another_key() -> None:
    """The separator stops ("a","bc") from colliding with ("ab","c")."""
    assert derive_seed("generation", "a", "bc") != derive_seed("generation", "ab", "c")


def test_seeds_stay_inside_the_signed_32_bit_range() -> None:
    seed = generation_seed(stage="C", task_id="x" * 200, sample_id="y" * 50, arm="M", turn=99)

    assert 0 <= seed < 2**31


def test_purposes_are_separated_and_validated() -> None:
    assert derive_seed("task_selection", "x") != derive_seed("run_order", "x")
    with_unknown = "not_a_purpose"
    try:
        derive_seed(with_unknown, "x")
    except ValueError as exc:
        assert "unknown seed purpose" in str(exc)
    else:  # pragma: no cover - the branch is the point of the test
        raise AssertionError("an unknown purpose must be rejected")


def test_stable_order_keys_are_orderable_and_distinct() -> None:
    keys = [stable_order_key("state_selection", "task", f"trace{i}", "0") for i in range(20)]

    assert len(set(keys)) == 20
    assert sorted(keys) == sorted(set(keys))


def test_a_negative_turn_is_rejected() -> None:
    try:
        generation_seed(stage="C", task_id="t", sample_id="s", arm="F", turn=-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a negative turn must be rejected")
