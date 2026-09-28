"""Run an agent arm with a PEFT adapter, driving the frozen runner rather than editing it.

Why this exists
---------------
The three-arm campaign needs two arms the stored runner cannot express: SFT+agent
(base snapshot + the SFT adapter) and SFT+GRPO r0+agent (merged SFT + the r0
adapter). The weights have to load exactly as the corresponding *direct* arms
loaded them. Otherwise the ``sft_direct`` <-> ``sft_agent`` contrast -- the one
this campaign exists to measure -- would carry a merge-rounding difference on top
of the channel difference it is supposed to isolate.

``scripts/campaign-20260926/rule_baseline.py`` cannot be given an ``--adapter``
flag. It writes ``git hash-object`` of its own committed file into every manifest
as ``router_source_sha256``, a blocking identity field, and the stored ``base+tool``
arm pinned that blob (``e591c886…``): one edit and every future agent arm stops
being pairable with the arm this campaign pairs against. (Same reason the merged
checkpoint is a worse answer here than an adapter -- it would add a merge the
direct arm never had.)

So this wrapper drives the frozen runner instead of changing it. It parses the
run's argv with the runner's own parser, patches exactly one call -- the model
loader, to pass ``adapter=`` -- hands the namespace to ``rule_baseline.run``, and
restores the loader afterwards. Everything the manifest records, including
``router_source_sha256``, is produced by the unmodified runner.

Order is deliberate. The identity is emitted **before** the model is loaded, so a
task set that is not the frozen one, a relative path, a missing adapter or a
shard mismatch costs a CPU second rather than a GPU hour. It lands beside the run
directory (``<output-dir>.identity.json``) because the runner refuses to start
unless its output directory is empty, and a copy is placed inside
``<output-dir>/identity.json`` once the run has finished, so the artifact carries
its own weights identity -- the frozen manifest has no field for it.

Usage
-----
::

    ADAPTIVE_MATH_SANDBOX_URL=http://localhost:8080 \\
    .venv/bin/python scripts/campaign-20260927/run_arm_with_adapter.py \\
        --mode all-tools --shard-id 0 --shard-count 3 \\
        --data <repo>/data/processed/v1/frozen_eval.parquet \\
        --task-ids-file <repo>/artifacts/eval/thesis_e0_base_direct_b1/task_ids.txt \\
        --model <base snapshot | merged SFT> \\
        --adapter <adapter dir> --adapter-kind {sft,rl} \\
        --agent-config <repo>/configs/agent/default.yaml \\
        --reward-config <repo>/configs/reward/r0.yaml \\
        --output-dir <repo>/artifacts/rollout_health/<arm>.shard0

``--adapter`` is required (an arm that loads none belongs to the plain runner),
and ``--dry-run`` stops after the identity, which makes the whole pre-flight one
command: emit, then gate, then run.

Exit codes: 0 ran (or dry-ran), 1 refused by the pre-flight, 2 usage error.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
RUN_ARM = REPO / "scripts" / "eval" / "emit_arm_identity.py"


def load_script(name: str, path: Path) -> ModuleType:
    """Import a sibling script, registering it before executing it.

    Registration is not optional: a module executed without a ``sys.modules``
    entry breaks ``from __future__ import annotations`` resolution at class
    creation, which is how this repository's scripts fail when loaded any other
    way.
    """
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


emitter = load_script("emit_arm_identity", RUN_ARM)
rule_baseline = emitter.runner

# One parser for the gate and the run, and the runner's own ``run`` untouched.
build_parser = emitter.build_parser
emit = emitter.emit


@contextmanager
def adapter_loader(adapter: Path) -> Iterator[None]:
    """Pass ``adapter=`` to the one call the runner makes to load weights.

    Scoped to a context manager on purpose: the runner is imported by everything
    else in this process, and a patch left behind would silently change what the
    *next* arm in the same interpreter loads. The original is read out of the
    class ``__dict__`` and put back there verbatim, so "restored" means the same
    object came back and not a bound method that merely behaves like it.
    """
    client = rule_baseline.TransformersModelClient
    original = client.__dict__["from_pretrained"]

    def patched(cls: type, model_id: str, **kwargs: object) -> object:
        return original.__func__(cls, model_id, adapter=str(adapter), **kwargs)

    client.from_pretrained = classmethod(patched)  # type: ignore[method-assign]
    try:
        yield
    finally:
        client.from_pretrained = original  # type: ignore[method-assign]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Not a general-purpose entry point: ``--adapter`` is what this file adds, and
    # an arm that needs none is the emitter's own business. Refusing here keeps
    # one invocation per case instead of two spellings of the base arm.
    if args.adapter is None:
        parser.error(
            "--adapter is required: arms that load no adapter run rule_baseline.py "
            "directly (the stored base arm was produced that way)"
        )

    output_dir = Path(args.output_dir)
    # Beside the run directory, never inside it: the runner refuses to start unless
    # its output directory is empty. ``--out`` overrides the path, and is honoured
    # rather than ignored -- a flag that quietly does nothing is worse than no flag.
    sidecar = (
        Path(args.out)
        if args.out is not None
        else output_dir.with_suffix(output_dir.suffix + ".identity.json")
    )
    try:
        identity = emit(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"PRE-FLIGHT FAIL: {exc}", file=sys.stderr)
        return 1
    # Before the model: a wrong task list, a relative path, a missing adapter or a
    # shard mismatch costs a second here instead of a GPU hour two lines below.
    sidecar.write_text(emitter.dumps(identity))
    print(
        f"PRE-FLIGHT OK: {identity['task_count']} frozen tasks, "
        f"model={identity['model']}, "
        f"adapter={identity['adapter_sha256'][:12]}… ({identity['adapter_kind']}), "
        f"identity at {sidecar}",
        file=sys.stderr,
    )
    print(
        "Gate it now: scripts/eval/verify_arm_identity.py --baseline "
        "<stored shard manifest> --candidate "
        f"{sidecar} --justify docs/results/campaign-2026-09-27/source-drift-justification.md",
        file=sys.stderr,
    )

    if args.dry_run:
        return 0

    with adapter_loader(Path(args.adapter)):
        asyncio.run(rule_baseline.run(args))

    # The runner owns its output directory and refuses to start unless it is
    # empty, so the identity is copied in only after a clean finish -- next to the
    # authoritative manifest.json, marked ``status: dry-run`` so the two cannot be
    # confused.
    shutil.copyfile(sidecar, output_dir / "identity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
