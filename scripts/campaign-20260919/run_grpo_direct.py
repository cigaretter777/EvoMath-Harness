"""Direct-launch wrapper for run_grpo.main (no torchrun; see register-center note)."""
import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path("/root/autodl-tmp/Adaptive-Solver-main-git")
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("run_grpo", REPO / "scripts" / "train" / "run_grpo.py")
    run_grpo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(run_grpo)
    return run_grpo.main(["--config", args.config])


if __name__ == "__main__":
    raise SystemExit(main())
