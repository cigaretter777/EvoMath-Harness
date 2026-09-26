#!/usr/bin/env bash
# Single source of truth for "does this container actually hold a GPU".
#
# Do not gate on nvidia-smi's exit code. On the AutoDL host /usr/bin/nvidia-smi is
# a 0-byte file: bash executes a file with no shebang as an empty shell script,
# which exits 0, so `if ! nvidia-smi; then wait` reads "card present" in a
# container with no card -- and a watchdog wired that way launches training into
# a cardless box while logging "GPU up" (observed 2026-09-24T16:34Z).
#
# The authority is /dev: character device nodes appear with the allocation and
# vanish without it, whereas /proc/driver/nvidia/gpus keeps listing the host's
# eight RTX 4090s even when this container holds none. nvidia-smi is recorded as
# advisory evidence and never decides.
#
# Exit status: 0 = usable card, non-zero = do not launch. One JSON object on
# stdout in both cases, so a caller can log the reason instead of guessing.
set -euo pipefail

devdir="${GPU_PROBE_DEV_DIR:-/dev}"

count_nodes() {
    # nvidiactl is the driver, not a card; nvidia0..n are the cards. A missing
    # devdir is "no card", not a probe error, so it must not trip pipefail.
    if [ ! -d "$devdir" ]; then echo 0; return 0; fi
    find "$devdir" -maxdepth 1 -type c -name 'nvidia[0-9]*' | wc -l | tr -d ' '
}

ctl_present=false
if [ -c "$devdir/nvidiactl" ]; then ctl_present=true; fi
gpu_count="$(count_nodes)"

smi_path="$(command -v nvidia-smi || true)"
smi_gpus=0
if [ -n "$smi_path" ] && [ -s "$smi_path" ]; then
    smi_gpus="$("$smi_path" -L 2>/dev/null | grep -c '^GPU ' || true)"
fi

usable=false
reason=""
if [ "$ctl_present" != true ]; then
    reason="no nvidiactl character device under $devdir"
elif [ "$gpu_count" -lt 1 ]; then
    reason="no nvidia<N> gpu character device under $devdir"
else
    usable=true
    reason="gpu device nodes present"
fi

printf '{"usable":%s,"reason":"%s","gpu_count":%s,"nvidia_smi_listed_gpus":%s,"dev_dir":"%s"}\n' \
    "$usable" "$reason" "$gpu_count" "$smi_gpus" "$devdir"

if [ "$usable" != true ]; then exit 1; fi
exit 0
