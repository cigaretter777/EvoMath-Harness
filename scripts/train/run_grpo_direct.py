"""Launch run_grpo.main() in-process, without torchrun.

This container hangs in torchrun's register-center handshake, so both GRPO runs
that ever completed here (R0 on 2026-09-19, R2 the same night) were started this
way. The wrapper loads scripts/train/run_grpo.py by path because that directory is
not an importable package, and forwards every remaining argument verbatim, so
``--dry-run`` and ``--output-dir`` behave exactly as they do under torchrun.

It used to live in artifacts/runs/, which is gitignored: the only launch path
verified on this host was therefore untracked, and the runbook still named only
the broken one.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

_RUN_GRPO = Path(__file__).resolve().parent / "run_grpo.py"


def main(argv: list[str] | None = None) -> int:
    spec = importlib.util.spec_from_file_location("run_grpo", _RUN_GRPO)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_RUN_GRPO}")
    run_grpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_grpo)
    return run_grpo.main(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
