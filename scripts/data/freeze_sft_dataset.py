"""Freeze a materialized SFT parquet into the 90/10 train/validation split.

The defaults are the released dp_v1 freeze, so a bare invocation stays what it
always was. The TIR cold start passes its own paths:

    uv run python scripts/data/freeze_sft_dataset.py \
        --input data/processed/sft_tir_v1_10k.parquet \
        --out-dir data/processed/sft_tir_v1 \
        --manifest data/manifests/sft_tir_v1_split.json \
        --report artifacts/reports/sft_tir_v1_quality_report.md \
        --dataset-version sft-tir-v1
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

SEED = 42

TRAIN_FRACTION = 0.9

DEFAULT_INPUT = Path("data/processed/sft_dp_v1_10k.parquet")

DEFAULT_OUT_DIR = Path("data/processed/sft_dp_v1")

DEFAULT_MANIFEST = Path("data/manifests/sft_dp_v1_split.json")

DEFAULT_REPORT = Path("artifacts/reports/sft_dp_v1_quality_report.md")

DEFAULT_DATASET_VERSION = "sft-dp-v1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)

    print("total:", len(df))

    # deterministic shuffle
    rng = random.Random(SEED)

    indices = list(range(len(df)))
    rng.shuffle(indices)

    split = int(len(df) * TRAIN_FRACTION)

    train_idx = indices[:split]
    val_idx = indices[split:]

    train = df.iloc[train_idx]
    val = df.iloc[val_idx]

    train_path = args.out_dir / "train.parquet"
    val_path = args.out_dir / "validation.parquet"

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    manifest = {

        "dataset_version":
            args.dataset_version,

        "source":
            str(args.input),

        "seed":
            SEED,

        "total":
            len(df),

        "train":
            {
                "count":len(train),
                "file":str(train_path),
                "sha256":sha256(train_path)
            },

        "validation":
            {
                "count":len(val),
                "file":str(val_path),
                "sha256":sha256(val_path)
            }
    }

    args.manifest.write_text(
        json.dumps(
            manifest,
            indent=2
        )
    )

    report = f"""
# SFT Dataset {args.dataset_version} Quality Report


## Dataset

dataset version:

{args.dataset_version}


source:

{args.input}


total:

{len(df)}


## Split


train:

{len(train)}


validation:

{len(val)}


seed:

{SEED}


## Hash


train:

{sha256(train_path)}


validation:

{sha256(val_path)}

"""

    args.report.write_text(report)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
