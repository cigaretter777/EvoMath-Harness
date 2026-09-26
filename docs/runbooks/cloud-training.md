# Cloud training runbook

## Provision and image

Use a Linux GPU host with the capacity in `configs/cloud/h100_2x.yaml`. Build
only from an immutable CUDA image digest; never substitute a tag.

The locked rollout stack is **Torch 2.8.0 + CUDA 12.8 wheels + vLLM 0.11.0**.
vLLM 0.11.0 requires Torch 2.8.0, so do not combine it with the older
Torch 2.6 / cu124 route. On AutoDL (where Docker is normally unavailable),
install the same lock in the data-disk Conda environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/adaptive-math
python -m pip install --force-reinstall \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install vllm==0.11.0
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Run `python -m pip check` and the GPU import check before starting SFT or
GRPO. Do not run these installations concurrently.

```bash
docker build -f docker/Dockerfile.train -t adaptive-math-rl:train \
  --build-arg CUDA_BASE_IMAGE='nvidia/cuda@sha256:<approved-digest>' .
```

Set `SANDBOXFUSION_IMAGE_DIGEST` and `SANDBOXFUSION_GIT_SHA`, then start the
separately pinned sandbox with `docker compose -f docker/compose.sandbox.yml up -d`.
Do not put cloud, Hugging Face, W&B, or S3 credentials in Git.

## Preflight and persistence

Copy the private task-pool JSONL and manifest, set `ADAPTIVE_MATH_TASK_MANIFEST`
and `ADAPTIVE_MATH_SANDBOX_URL`, then run `scripts/cloud/preflight.sh`. Start a
`tmux` session before training so SSH loss cannot terminate it.

`preflight.py --require-sandbox` probes the sandbox twice: `GET /v1/ping` must
answer `pong`, then a real `POST /run_code` of `print(6 * 7)` must return
`stdout` `42\n`. SandboxFusion serves no `/health` route, so do not "fix" a
failing probe by pointing it at one. A ping that succeeds while the execution
probe fails means the HTTP front is up but the executor is not.

Capacity thresholds default to the cloud H100 host. A single-GPU box with a
smaller data disk must lower them explicitly rather than editing the script:

```bash
export ADAPTIVE_MATH_MIN_DISK_GB=100   # free GiB required in the run directory
export ADAPTIVE_MATH_MIN_RAM_GB=64
```

On AutoDL there is no Docker, so `compose.sandbox.yml` cannot be used and the
sandbox arrives as an SSH remote forward on `127.0.0.1:8080` from the host that
runs SandboxFusion. Nothing in this repository keeps that tunnel alive: after a
container restart the port refuses connections until the peer reconnects, and
every Python tool call in that window degrades to `unavailable`. Re-run
`preflight.sh` after any restart, and treat a mid-run drop in tool successes as
a tunnel failure before blaming the policy.

```bash
tmux new -s adaptive-math
scripts/cloud/launch_grpo.sh --dry-run
scripts/cloud/launch_grpo.sh
```

`launch_sft.sh` and `launch_grpo.sh` both run `preflight.sh` themselves, so a
missing manifest, GPU, sandbox or backend contract stops the run before any GPU
minute is spent. SFT sets `ADAPTIVE_MATH_REQUIRE_SANDBOX=0` because it trains on
frozen trajectories and never calls a tool; GRPO may not opt out.

Run `sync_artifacts.sh` after every complete checkpoint. It writes checksums
and copies the run evidence to the configured S3-compatible prefix.

## Emergency stop

Run `python scripts/cloud/stop_if_unhealthy.py --metrics <run>/metrics.jsonl`
between evaluation intervals. On a non-zero result, stop the trainer, sync
evidence, stop Ray and SandboxFusion, then terminate the GPU instance. Record
the reason and direct cost. Resume only through `resume_latest.sh` after the
resolved config and input hashes match.
