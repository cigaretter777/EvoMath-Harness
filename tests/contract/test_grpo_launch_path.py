"""The launch path that actually worked must be the launch path in the repo.

Both successful GRPO runs on this box were started with
``python scripts/train/run_grpo.py``-as-a-module through a wrapper that lived in
``artifacts/runs/`` -- a gitignored directory -- because torchrun hangs on this
container at the register-center handshake. ``launch_grpo.sh`` still named only
torchrun, so the runbook path was both unverified and, on this host, broken.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
DIRECT = REPO_ROOT / "scripts" / "train" / "run_grpo_direct.py"


def test_direct_launcher_is_tracked_and_does_not_bake_in_a_checkout_path() -> None:
    """A wrapper that hardcodes one absolute checkout cannot be replayed from a
    clean clone, which is the whole point of keeping it in the repo. The previous
    copy of this script lived under artifacts/, i.e. it was never tracked at all."""
    source = DIRECT.read_text()

    assert "Path(__file__).resolve().parents[2]" in source
    assert "run_grpo.py" in source
    assert "/root/autodl-tmp" not in source


def test_launch_grpo_defaults_to_the_verified_direct_path() -> None:
    source = (REPO_ROOT / "scripts" / "cloud" / "launch_grpo.sh").read_text()

    assert "run_grpo_direct.py" in source
    assert "ADAPTIVE_MATH_USE_TORCHRUN" in source


def test_launch_grpo_still_gates_on_preflight_in_both_modes() -> None:
    """Choosing an engine is not a licence to skip the sandbox/GPU gate."""
    source = (REPO_ROOT / "scripts" / "cloud" / "launch_grpo.sh").read_text()
    gate = source.index("scripts/cloud/preflight.sh")
    engine = max(source.index("run_grpo_direct.py"), source.index("torchrun"))

    assert gate < engine
