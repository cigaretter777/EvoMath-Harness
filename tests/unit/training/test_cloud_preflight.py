import importlib.util
import json
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).parents[3]
_spec = importlib.util.spec_from_file_location(
    "cloud_preflight", REPO_ROOT / "scripts" / "cloud" / "preflight.py"
)
cloud_preflight = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cloud_preflight)


def test_preflight_dry_run_validates_immutable_inputs_without_gpu(tmp_path: Path) -> None:
    manifest = tmp_path / "tasks.manifest.json"
    manifest.write_text(json.dumps({"schema_version": "sft-task-pool-v1"}))
    upstreams = tmp_path / "upstreams.json"
    upstreams.write_text((REPO_ROOT / "third_party" / "manifest.json").read_text())

    report = cloud_preflight.preflight(
        task_manifest=manifest,
        upstream_manifest=upstreams,
        require_gpu=False,
        require_sandbox=False,
    )

    assert report["ok"] is True
    assert report["inputs"]["task_manifest"]["sha256"]
    assert report["gpu"]["required"] is False


def test_preflight_probes_the_sandbox_and_fails_when_it_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting the URL is not evidence the sandbox works: preflight must actually
    reach it, because a dead tunnel otherwise silently turns every Python tool
    call into UNAVAILABLE for the whole run."""
    manifest = tmp_path / "tasks.manifest.json"
    manifest.write_text("{}")
    # Port 1 on loopback refuses instantly; nothing is listening there.
    monkeypatch.setenv("ADAPTIVE_MATH_SANDBOX_URL", "http://127.0.0.1:1")

    with pytest.raises(RuntimeError, match="sandbox"):
        cloud_preflight.preflight(
            task_manifest=manifest,
            upstream_manifest=REPO_ROOT / "third_party" / "manifest.json",
            require_gpu=False,
            require_sandbox=True,
        )


def test_preflight_records_sandbox_evidence_when_the_probe_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, respx_mock: object
) -> None:
    manifest = tmp_path / "tasks.manifest.json"
    manifest.write_text("{}")
    monkeypatch.setenv("ADAPTIVE_MATH_SANDBOX_URL", "http://sandbox.test")
    respx_mock.get("http://sandbox.test/v1/ping").mock(  # type: ignore[attr-defined]
        return_value=httpx.Response(200, json="pong")
    )
    respx_mock.post("http://sandbox.test/run_code").mock(  # type: ignore[attr-defined]
        return_value=httpx.Response(
            200,
            json={
                "status": "Success",
                "message": "",
                "compile_result": None,
                "run_result": {
                    "status": "Finished",
                    "execution_time": 0.09,
                    "return_code": 0,
                    "stdout": "42\n",
                    "stderr": "",
                },
            },
        )
    )

    report = cloud_preflight.preflight(
        task_manifest=manifest,
        upstream_manifest=REPO_ROOT / "third_party" / "manifest.json",
        require_gpu=False,
        require_sandbox=True,
    )

    assert report["sandbox"] == {
        "required": True,
        "url": "http://sandbox.test",
        "reachable": True,
        "probe": "pong+42",
    }


def test_preflight_rejects_gpu_required_without_nvidia_smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "tasks.manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(cloud_preflight.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="nvidia-smi"):
        cloud_preflight.preflight(
            task_manifest=manifest,
            upstream_manifest=REPO_ROOT / "third_party" / "manifest.json",
            require_gpu=True,
            require_sandbox=False,
        )
