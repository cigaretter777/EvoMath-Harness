"""Validate immutable cloud-training inputs before renting a GPU."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from adaptive_math.core.hashing import sha256_hex
from adaptive_math.training.upstreams import validate_manifest

# The run probe must outlast the sandbox's own 10s default run_timeout, or a
# healthy-but-slow executor would be reported as dead.
_PROBE_TIMEOUT_SECONDS = 5.0
_PROBE_RUN_TIMEOUT_SECONDS = 15.0


def _file_evidence(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"required file does not exist: {path}")
    return {"path": str(path), "sha256": sha256_hex(path.read_bytes())}


def _gpu_evidence(required: bool, dev_dir: Path = Path("/dev")) -> dict[str, Any]:
    """Device nodes decide; nvidia-smi only annotates.

    /usr/bin/nvidia-smi is a 0-byte file on this host, and bash executes a file
    with no shebang as an empty shell script that exits 0, so a gate on its status
    reads "card present" in a container with no card. /proc/driver/nvidia/gpus is
    worse: it keeps listing the host's eight RTX 4090s when this container holds
    none. The character nodes under dev_dir track the allocation. The same rule
    lives in scripts/cloud/gpu_probe.sh for the shell pollers, which cannot afford
    a Python import per tick.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not required:
        return {"required": False, "nvidia_smi": nvidia_smi}

    listing: list[str] = []
    if nvidia_smi is None:
        smi_note = "absent from PATH"
    else:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            # ENOEXEC from the 0-byte stub. A crash here used to escape as a
            # traceback out of the gate that launch_grpo.sh depends on.
            smi_note = f"unexecutable: {exc}"
        else:
            listing = result.stdout.splitlines()
            smi_note = (
                f"listed {len(listing)}" if listing else f"no output (rc={result.returncode})"
            )

    ctl = dev_dir / "nvidiactl"
    nodes = sorted(path.name for path in dev_dir.glob("nvidia[0-9]*"))
    minimum = int(os.environ.get("ADAPTIVE_MATH_MIN_GPU_COUNT", "1"))
    if not ctl.is_char_device():
        raise RuntimeError(
            f"no usable GPU: {ctl} is not a character device (nvidia-smi: {smi_note})"
        )
    if len(nodes) < minimum:
        raise RuntimeError(
            f"no usable GPU: {len(nodes)} nvidia<N> node(s) under {dev_dir}, "
            f"ADAPTIVE_MATH_MIN_GPU_COUNT={minimum} (nvidia-smi: {smi_note})"
        )
    return {
        "required": True,
        "device_nodes": nodes,
        "devices": listing,
        "nvidia_smi": nvidia_smi,
        "nvidia_smi_note": smi_note,
    }


def _sandbox_evidence(required: bool) -> dict[str, Any]:
    """A configured URL is not evidence the sandbox works. Probe liveness and one
    real execution: a dead sandbox degrades every Python tool call to UNAVAILABLE
    for the whole run, and the reward then trains against a broken tool."""
    if not required:
        return {"required": False}
    url = os.environ.get("ADAPTIVE_MATH_SANDBOX_URL")
    if not url:
        raise RuntimeError("ADAPTIVE_MATH_SANDBOX_URL is required when sandbox checks are enabled")
    base = url.rstrip("/")
    try:
        ping = httpx.get(f"{base}/v1/ping", timeout=_PROBE_TIMEOUT_SECONDS)
        if ping.status_code != 200 or ping.json() != "pong":
            raise RuntimeError(f"sandbox ping at {base}/v1/ping returned {ping.status_code}")
        run = httpx.post(
            f"{base}/run_code",
            json={"code": "print(6 * 7)", "language": "python"},
            timeout=_PROBE_RUN_TIMEOUT_SECONDS,
        )
        payload = run.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"sandbox at {base} is unreachable: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"sandbox at {base} answered with a non-JSON body") from exc
    result = payload.get("run_result") or {}
    if run.status_code != 200 or payload.get("status") != "Success" or result.get("stdout") != "42\n":
        raise RuntimeError(f"sandbox execution probe failed at {base}: {str(payload)[:300]}")
    return {"required": True, "url": url, "reachable": True, "probe": "pong+42"}


def preflight(
    *,
    task_manifest: Path,
    upstream_manifest: Path,
    require_gpu: bool,
    require_sandbox: bool,
    dev_dir: Path = Path("/dev"),
) -> dict[str, Any]:
    """Return evidence or fail before model loading and cloud cost are incurred."""
    task_evidence = _file_evidence(task_manifest)
    upstream_evidence = _file_evidence(upstream_manifest)
    try:
        validate_manifest(json.loads(upstream_manifest.read_text()))
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid upstream manifest: {exc}") from exc
    return {
        "ok": True,
        "inputs": {"task_manifest": task_evidence, "upstream_manifest": upstream_evidence},
        "gpu": _gpu_evidence(require_gpu, dev_dir),
        "sandbox": _sandbox_evidence(require_sandbox),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--upstream-manifest", type=Path, default=Path("third_party/manifest.json"))
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-sandbox", action="store_true")
    parser.add_argument("--dev-dir", type=Path, default=Path("/dev"))
    args = parser.parse_args()
    try:
        report = preflight(
            task_manifest=args.task_manifest,
            upstream_manifest=args.upstream_manifest,
            require_gpu=args.require_gpu,
            require_sandbox=args.require_sandbox,
            dev_dir=args.dev_dir,
        )
    except RuntimeError as exc:
        # One line, not a traceback: this is the gate launch_grpo.sh stops at, and
        # the reason has to be readable at a glance.
        print(f"preflight FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
