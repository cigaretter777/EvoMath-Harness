"""The GPU gate must never trust nvidia-smi's exit code.

``/usr/bin/nvidia-smi`` on this AutoDL host is a 0-byte file. bash runs a file
with no shebang as an empty shell script, which exits 0, so the gate written in
``campaign_monitor.sh`` -- ``if ! nvidia-smi; then wait`` -- reads "card present"
on a container that has no card at all, and a watchdog wired that way launches
GRPO into a cardless box while logging "GPU up".

Device nodes are the container-level truth instead: ``/proc/driver/nvidia/gpus``
lists the host's eight RTX 4090s even with nothing allocated here, while
``/dev/nvidia*`` is absent.
"""

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
PROBE = REPO_ROOT / "scripts" / "cloud" / "gpu_probe.sh"


def _run_probe(devdir: Path, extra_path: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["GPU_PROBE_DEV_DIR"] = str(devdir)
    if extra_path is not None:
        env["PATH"] = f"{extra_path}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", str(PROBE)], capture_output=True, text=True, env=env, timeout=60, check=False
    )


def _fake_card(devdir: Path, gpu_count: int = 1) -> None:
    """Create the real character device nodes the probe looks for.

    ``mknod`` needs root or CAP_MKNOD; the GitHub Actions runner has neither,
    so these cases skip there. The AutoDL host (root) still exercises them —
    that is where the false-positive this file guards against was observed.
    """
    devdir.mkdir(parents=True, exist_ok=True)
    try:
        ctl = devdir / "nvidiactl"
        if not ctl.exists():
            os.mknod(str(ctl), stat.S_IFCHR | 0o666, os.makedev(195, 255))
        for index in range(gpu_count):
            node = devdir / f"nvidia{index}"
            if not node.exists():
                os.mknod(str(node), stat.S_IFCHR | 0o666, os.makedev(195, index))
    except PermissionError as e:
        pytest.skip(f"creating char device nodes needs root/CAP_MKNOD: {e}")


def _zero_byte_nvidia_smi(bindir: Path) -> Path:
    """Reproduce the host's stub byte for byte: empty, executable."""
    bindir.mkdir(parents=True, exist_ok=True)
    stub = bindir / "nvidia-smi"
    stub.write_text("")
    stub.chmod(0o755)
    return stub


def test_probe_blocks_when_the_container_has_no_gpu_device_nodes(tmp_path: Path) -> None:
    proc = _run_probe(tmp_path / "dev")

    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["usable"] is False
    assert "nvidiactl" in payload["reason"]


def test_probe_reports_a_attached_card_as_usable(tmp_path: Path) -> None:
    devdir = tmp_path / "dev"
    _fake_card(devdir, gpu_count=1)

    proc = _run_probe(devdir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["usable"] is True
    assert payload["gpu_count"] == 1


def test_probe_blocks_on_a_partial_attach_without_any_gpu_node(tmp_path: Path) -> None:
    """nvidiactl alone is not a card: the driver is bound but nothing is allocated."""
    devdir = tmp_path / "dev"
    _fake_card(devdir, gpu_count=0)

    proc = _run_probe(devdir)

    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["usable"] is False
    assert "gpu" in payload["reason"].lower()


def test_zero_byte_nvidia_smi_cannot_make_the_probe_report_a_card(tmp_path: Path) -> None:
    """The exact false positive seen at 2026-09-24T16:34Z on a cardless container."""
    devdir = tmp_path / "dev"
    devdir.mkdir()
    _zero_byte_nvidia_smi(tmp_path / "bin")

    proc = _run_probe(devdir, extra_path=tmp_path / "bin")

    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["usable"] is False


def test_zero_byte_nvidia_smi_does_not_block_a_real_card(tmp_path: Path) -> None:
    """Fail-closed must not mean fail-forever: the stub is not the authority."""
    devdir = tmp_path / "dev"
    _fake_card(devdir, gpu_count=2)
    _zero_byte_nvidia_smi(tmp_path / "bin")

    proc = _run_probe(devdir, extra_path=tmp_path / "bin")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["usable"] is True
    assert payload["gpu_count"] == 2


def test_probe_records_the_nvidia_smi_verdict_without_deciding_with_it(tmp_path: Path) -> None:
    devdir = tmp_path / "dev"
    _fake_card(devdir, gpu_count=1)
    _zero_byte_nvidia_smi(tmp_path / "bin")

    payload = json.loads(_run_probe(devdir, extra_path=tmp_path / "bin").stdout)

    assert payload["nvidia_smi_listed_gpus"] == 0
    assert payload["usable"] is True


def test_the_polling_watchdogs_gate_through_the_probe() -> None:
    """The two scripts that decide on their own whether to launch a GPU job used to
    test `if ! nvidia-smi`, and the auto-launching one was the one that was wrong."""
    pollers = (
        REPO_ROOT / "scripts" / "campaign-20260924" / "campaign_monitor.sh",
        REPO_ROOT / "scripts" / "campaign-20260919" / "r2_rollout_health_watchdog.sh",
    )
    for path in pollers:
        source = path.read_text()
        assert "gpu_probe.sh" in source, f"{path.name} must gate through the shared probe"
        assert "! nvidia-smi" not in source, f"{path.name} still trusts nvidia-smi's exit code"


def test_preflight_delegates_the_gpu_gate_rather_than_reimplementing_it() -> None:
    """preflight.py owns the launch-time GPU decision. A second nvidia-smi call in
    the shell is the same drift the sandbox probe is already forbidden to have."""
    source = (REPO_ROOT / "scripts" / "cloud" / "preflight.sh").read_text()

    assert "nvidia-smi" not in source
    assert "--require-gpu" in source
    assert "preflight.py" in source


def test_probe_and_preflight_reach_the_same_verdict_on_this_host(tmp_path: Path) -> None:
    """Two implementations of one rule must not disagree, or the poller launches
    what preflight then refuses (or vice versa) and the night is lost either way."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cloud_preflight", REPO_ROOT / "scripts" / "cloud" / "preflight.py"
    )
    cloud_preflight = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cloud_preflight)

    probe = _run_probe(Path("/dev"))
    try:
        cloud_preflight._gpu_evidence(True)
        preflight_usable = True
    except RuntimeError:
        preflight_usable = False

    assert (probe.returncode == 0) is preflight_usable
