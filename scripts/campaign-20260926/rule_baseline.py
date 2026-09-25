"""Fixed-rule strategy baseline (thesis E0c) and Base+tool arm (thesis E0b).

Modes:

  --mode rule      Fixed-rule strategy. Each task is routed by deterministic
                   text/type rules into one of three channels:
                     sympy   -> agent rollout with ONLY the sympy tool
                     python  -> agent rollout with ONLY the python tool
                     direct  -> no rollout; the channel reuses the stored
                                base-direct eval arm (recorded in routing.jsonl)
                   This is the "固定规则策略" no-training baseline: the rule
                   decides which tool (if any) the model may use.

  --mode all-tools Base with unrestricted tools (thesis E0b). All tasks are
                   rolled out with both tools available.

Both modes: greedy decoding (temperature 0 -> do_sample=False), batch 1,
frozen Omni-MATH 200 (the rl_r0_200 pool, same task_ids as every stored arm),
production agent prompt, strict parser, live sandbox, reward config r0
(no tool-cost / invalid penalties: a no-training baseline must not bake in
an RL reward structure).

Why a separate script instead of extending run_rollout_health.py: that runner
always builds the two-tool registry and samples a subset with a seed; this
baseline needs per-task tool whitelists, the exact 200-task pool, and a
routing table that must be committed before the run so the git_sha in the
manifest pins the rules that produced it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import yaml

from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.loop import AgentLoop
from adaptive_math.agent.model_client import GenerationConfig, ModelClient
from adaptive_math.agent.transformers_client import TransformersModelClient
from adaptive_math.core.types import Budget, LabeledMathTask, MathTask
from adaptive_math.reward import RewardConfig
from adaptive_math.tools.base import Tool
from adaptive_math.tools.python_tool import PythonTool
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.tools.sandboxfusion import SandboxFusionClient
from adaptive_math.tools.sympy_tool import SympyTool
from adaptive_math.training.reward_bridge import reward_for_trajectory

REPO = Path(__file__).resolve().parents[2]

# --- router rules (v1, deterministic; see preregistration) ------------------

# Equation/symbolic markers. `[a-z]=` must not match answer blanks: in this
# pool the blank is always "n=$ \\qquad" / "m=$", i.e. the variable equals a
# dollar sign, so require the next char to be neither $ nor whitespace.
SYMPY_RE = re.compile(
    r"\bsolve\b|\bequations?\b|\broots?\s+of\b|=\s*0|[a-z]=\s*(?=[^$\s])",
    re.IGNORECASE,
)
PYTHON_RE = re.compile(
    r"\bdigits?\b|\bsum\b|\bproduct\b|\bremainder\b|\bmod\b|\d{4,}|\^{|!",
    re.IGNORECASE,
)
PYTHON_TYPES = {"Combinatorics", "Number Theory"}


def route(task: MathTask) -> str:
    text = task.problem
    problem_type = (task.metadata or {}).get("problem_type", "")
    if SYMPY_RE.search(text):
        return "sympy"
    if problem_type in PYTHON_TYPES or PYTHON_RE.search(text):
        return "python"
    return "direct"


# --- plumbing ---------------------------------------------------------------


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def router_source_sha256() -> str:
    """Hash of this file as committed, so the manifest pins the router rules."""
    return subprocess.run(
        ["git", "hash-object", str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def load_pool(path: Path) -> list[LabeledMathTask]:
    tasks: list[LabeledMathTask] = []
    with path.open() as handle:
        for line in handle:
            tasks.append(LabeledMathTask.model_validate(json.loads(line)))
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("rule", "all-tools"),
        default="rule",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=REPO / "artifacts/task_pools/rl_r0_200.jsonl",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/root/autodl-tmp/hf-cache/models--Qwen--Qwen3-1.7B/"
            "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        ),
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=REPO / "configs/agent/default.yaml",
    )
    parser.add_argument(
        "--reward-config",
        type=Path,
        default=REPO / "configs/reward/r0.yaml",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the routing table and exit without touching the GPU",
    )
    return parser


async def rollout(
    task: LabeledMathTask,
    budget: Budget,
    registry: ToolRegistry,
    model: ModelClient,
    generation: GenerationConfig,
    reward_config: RewardConfig,
    channel: str,
    sample_index: int,
) -> dict[str, Any]:
    trace_id = f"rule-baseline:{task.task.task_id}:{sample_index}"
    environment = OfflineMathEnv(task, budget, registry, trace_id=trace_id)
    started = time.perf_counter()
    trajectory = await AgentLoop().run(environment, model, generation)
    rollout_seconds = time.perf_counter() - started

    verdict = environment.evaluate()
    reward = 0.0
    reward_payload = None
    if verdict is not None:
        breakdown = reward_for_trajectory(
            trajectory,
            verdict,
            budget,
            max_generated_tokens=budget.max_steps * generation.max_new_tokens,
            config=reward_config,
        )
        reward = float(breakdown.total)
        reward_payload = breakdown.model_dump(mode="json")

    return {
        "sample_index": sample_index,
        "task_id": task.task.task_id,
        "trace_id": trace_id,
        "channel": channel,
        "reward": reward,
        "rollout_seconds": rollout_seconds,
        "trajectory": trajectory.model_dump(mode="json"),
        "verdict": verdict.model_dump(mode="json") if verdict is not None else None,
        "reward_breakdown": reward_payload,
    }


async def run(args: argparse.Namespace) -> None:
    pool_path = Path(args.pool)
    agent_config_path = Path(args.agent_config)
    reward_config_path = Path(args.reward_config)
    output_dir = Path(args.output_dir)

    if not pool_path.is_file():
        raise FileNotFoundError(pool_path)
    if not agent_config_path.is_file():
        raise FileNotFoundError(agent_config_path)
    if not reward_config_path.is_file():
        raise FileNotFoundError(reward_config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}. "
            "Use a different --output-dir or remove the partial run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    budget = Budget.model_validate(yaml.safe_load(agent_config_path.read_text()))
    reward_config = RewardConfig.model_validate(
        yaml.safe_load(reward_config_path.read_text())
    )

    tasks = load_pool(pool_path)
    routing = {task.task.task_id: route(task.task) for task in tasks}
    distribution = Counter(routing.values())

    print(
        f"[rule-baseline] mode={args.mode} routing distribution: "
        f"{dict(sorted(distribution.items()))}",
        flush=True,
    )

    if args.dry_run:
        for task in tasks:
            print(
                f"  {routing[task.task.task_id]:8s} {task.task.task_id}  "
                f"{task.task.problem[:90]!r}"
            )
        return

    # Everything below is the GPU section. A dry-run must not reach it.

    sandbox = SandboxFusionClient()
    registries = {
        "sympy": ToolRegistry([cast(Tool, SympyTool())]),
        "python": ToolRegistry([cast(Tool, PythonTool(sandbox))]),
        "both": ToolRegistry(
            [cast(Tool, SympyTool()), cast(Tool, PythonTool(sandbox))]
        ),
    }

    print(f"[rule-baseline] loading model: {args.model}", flush=True)
    model = TransformersModelClient.from_pretrained(str(args.model))
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    trajectories_path = output_dir / "trajectories.jsonl"
    _write_json(
        output_dir / "manifest.json",
        {
            "status": "running",
            "mode": args.mode,
            "pool": str(pool_path),
            "model": str(args.model),
            "agent_config": str(agent_config_path),
            "reward_config": str(reward_config_path),
            "task_count": len(tasks),
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
            "sandbox_url": os.environ.get(
                "ADAPTIVE_MATH_SANDBOX_URL", "http://127.0.0.1:8080"
            ),
            "git_sha": git_sha(),
            "router_source_sha256": router_source_sha256(),
            "routing_distribution": dict(sorted(distribution.items())),
        },
    )

    with (output_dir / "routing.jsonl").open("w") as handle:
        for task in tasks:
            handle.write(
                json.dumps(
                    {
                        "task_id": task.task.task_id,
                        "channel": routing[task.task.task_id],
                    }
                )
                + "\n"
            )

    correct_count = 0
    tool_calls_total = 0
    invalid_actions_total = 0
    generated_tokens_total = 0
    rolled_out = 0

    for task_index, task in enumerate(tasks, start=1):
        task_id = task.task.task_id
        channel = routing[task_id] if args.mode == "rule" else "both"

        if args.mode == "rule" and channel == "direct":
            # Reuses the stored base-direct arm; nothing to generate.
            continue

        print(
            f"[rule-baseline] task {task_index}/{len(tasks)} "
            f"id={task_id} channel={channel}",
            flush=True,
        )
        record = await rollout(
            task,
            budget,
            registries[channel],
            model,
            generation,
            reward_config,
            channel,
            0,
        )
        _append_jsonl(trajectories_path, record)
        rolled_out += 1

        usage = record["trajectory"]["usage"]
        tool_calls_total += usage["tool_calls"]
        invalid_actions_total += usage["invalid_actions"]
        generated_tokens_total += usage["generated_tokens"]
        if record["verdict"] is not None and record["verdict"]["reward"] > 0:
            correct_count += 1

        print(
            f"  reward={record['reward']:.3f} "
            f"termination={record['trajectory']['termination_reason']} "
            f"tools={usage['tool_calls']} tokens={usage['generated_tokens']} "
            f"({record['rollout_seconds']:.0f}s)",
            flush=True,
        )

    summary = {
        "mode": args.mode,
        "task_count": len(tasks),
        "rolled_out": rolled_out,
        "direct_reused": len(tasks) - rolled_out,
        "correct_count": correct_count,
        "tool_calls_total": tool_calls_total,
        "invalid_actions_total": invalid_actions_total,
        "generated_tokens_total": generated_tokens_total,
        "routing_distribution": dict(sorted(distribution.items())),
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "COMPLETE").write_text("")
    print(f"[rule-baseline] complete: {summary}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
