"""Static contracts for scripts that are exercised only on a Linux GPU host."""

from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_cloud_shell_scripts_are_strict_and_support_dry_run() -> None:
    names = (
        "preflight.sh",
        "launch_sft.sh",
        "launch_grpo.sh",
        "resume_latest.sh",
        "sync_artifacts.sh",
    )
    for name in names:
        source = (REPO_ROOT / "scripts" / "cloud" / name).read_text()
        assert "set -euo pipefail" in source
        assert "--dry-run" in source


def test_training_image_requires_an_immutable_cuda_base_and_pinned_upstreams() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.train").read_text()
    lock = (REPO_ROOT / "docker" / "requirements.cuda.lock").read_text()

    assert "ARG CUDA_BASE_IMAGE" in dockerfile
    assert "FROM ${CUDA_BASE_IMAGE}" in dockerfile
    assert "20bd331bdbc9026a5668e11362178e10ab7400c8" in dockerfile
    assert "cu128" in dockerfile
    assert "torch==2.8.0" in lock
    assert "torchvision==0.23.0" in lock
    assert "torchaudio==2.8.0" in lock
    assert "vllm==0.11.0" in lock
    assert "flash-attn==" in lock


def test_preflight_probes_the_sandbox_through_one_source_of_truth() -> None:
    """SandboxFusion serves /v1/ping, not /health, and preflight.py already runs a
    two-level probe (ping + a real execution). A second curl probe in the shell can
    only drift from it, so the shell must stay delegated."""
    source = (REPO_ROOT / "scripts" / "cloud" / "preflight.sh").read_text()

    assert "/health" not in source
    assert "--require-sandbox" in source
    assert "preflight.py" in source


def test_launch_scripts_gate_on_preflight_before_spending_gpu_time() -> None:
    for name in ("launch_sft.sh", "launch_grpo.sh"):
        source = (REPO_ROOT / "scripts" / "cloud" / name).read_text()
        assert "preflight.sh" in source, f"{name} must run preflight before training"


def test_runbook_documents_the_disk_threshold_override_for_single_gpu_hosts() -> None:
    """The 500GB default fits the cloud H100 host; a single 24GB-GPU box has far
    less free space, so the override must be written down or preflight blocks
    every local run."""
    runbook = (REPO_ROOT / "docs" / "runbooks" / "cloud-training.md").read_text()

    assert "ADAPTIVE_MATH_MIN_DISK_GB" in runbook


def test_runbook_covers_persistence_and_emergency_stop() -> None:
    runbook = (REPO_ROOT / "docs" / "runbooks" / "cloud-training.md").read_text()

    assert "tmux" in runbook
    assert "Emergency stop" in runbook
    assert "SANDBOXFUSION_IMAGE_DIGEST" in runbook
