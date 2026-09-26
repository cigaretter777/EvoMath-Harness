"""Export a verl full-model checkpoint's inline LoRA into a PEFT adapter dir.

Campaign tool (2026-09-19): verl saves lora_A/lora_B inline in the full FSDP
state dict (base_model.model.model...). Extract them into adapter_model.safetensors
+ adapter_config.json so run_model_eval.py --adapter can load it.
"""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import save_file

LORA_A = ".lora_A.default.weight"
LORA_B = ".lora_B.default.weight"
LORA_A_STRIPPED = ".lora_A.weight"
LORA_B_STRIPPED = ".lora_B.weight"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    args = ap.parse_args()

    sd = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    assert isinstance(sd, dict), f"unexpected checkpoint root type: {type(sd)}"
    lora_weights = {
        # verl saves FSDP-wrapped PEFT keys with the adapter-name suffix
        # (lora_A.default.weight); PEFT's save_pretrained format has no
        # suffix (lora_A.weight). Strip it so PeftModel loads the deltas.
        key.replace(".lora_A.default.weight", ".lora_A.weight").replace(
            ".lora_B.default.weight", ".lora_B.weight"
        ): sd[key]
        for key in sd
        if key.endswith(LORA_A) or key.endswith(LORA_B)
    }
    if not lora_weights:
        raise SystemExit("no lora_A/lora_B weights found in checkpoint")
    target_modules = sorted(
        {
            key.split(".self_attn.")[1].split(".")[0]
            if ".self_attn." in key
            else key.split(".mlp.")[1].split(".")[0]
            for key in lora_weights
            if key.endswith(LORA_A_STRIPPED)
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(lora_weights, args.out / "adapter_model.safetensors")
    config = {
        "base_model_name_or_path": args.base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": args.alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": args.rank,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    (args.out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    (args.out / "rl_provenance.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "exported_at_utc": datetime.now(UTC).isoformat(),
                "lora_rank": args.rank,
                "lora_alpha": args.alpha,
            },
            indent=2,
        )
    )
    (args.out / "COMPLETE").write_text("")
    print(
        f"exported {len(lora_weights)} lora tensors -> {args.out}\n"
        f"target_modules={target_modules}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
