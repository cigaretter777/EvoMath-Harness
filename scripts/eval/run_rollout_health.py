"""Run a bounded real-agent rollout-health gate before GRPO training."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.loop import AgentLoop
from adaptive_math.agent.model_client import GenerationConfig
from adaptive_math.agent.parser import TOLERANCE_RULES
from adaptive_math.agent.transformers_client import TransformersModelClient
from adaptive_math.core.types import Budget
from adaptive_math.reward import RewardConfig
from adaptive_math.tools.base import Tool
from adaptive_math.tools.python_tool import PythonTool
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.tools.sandboxfusion import SandboxFusionClient
from adaptive_math.tools.sympy_tool import SympyTool
from adaptive_math.training.reward_bridge import (
    group_advantages,
    reward_for_trajectory,
)
from adaptive_math.training.rollout_health import (
    row_to_labeled_task,
    select_task_rows,
    summarize_group_rewards,
)


def _parse_tolerance_rules(value: str) -> tuple[str, ...]:
    rules = tuple(rule for rule in value.split(",") if rule)
    unknown = sorted(set(rules) - set(TOLERANCE_RULES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown parser-tolerance rule(s) {unknown}; valid: {TOLERANCE_RULES}"
        )
    return rules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real AgentLoop rollout-health evaluation before GRPO."
    )

    parser.add_argument(
        "--data",
        default="data/processed/v1/rl_dev.parquet",
    )
    parser.add_argument(
        "--model",
        default="artifacts/models/qwen3_1_7b_sft_dp_v1_merged",
    )
    parser.add_argument(
        "--agent-config",
        default="configs/agent/default.yaml",
    )
    parser.add_argument(
        "--reward-config",
        default="configs/reward/r0.yaml",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="optional PEFT adapter directory to wrap the base model",
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--parser-tolerance",
        type=_parse_tolerance_rules,
        default=None,
        metavar="RULES",
        help="comma-separated parser tolerance rules to enable "
        "(unclosed_think,bare_final_scalar); default: strict parser",
    )

    return parser


def default_output_dir(args: argparse.Namespace) -> Path:
    return Path(
        "artifacts/rollout_health/"
        f"smoke_{args.task_count}x{args.group_size}_seed{args.selection_seed}"
    )


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path}. "
            "Use a different --output-dir or remove the partial run."
        )
    path.mkdir(parents=True, exist_ok=True)


def _load_budget(path: Path) -> Budget:
    raw = yaml.safe_load(path.read_text())
    return Budget.model_validate(raw)


def _load_reward(path: Path) -> RewardConfig:
    raw = yaml.safe_load(path.read_text())
    return RewardConfig.model_validate(raw)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)


async def run(args: argparse.Namespace) -> Path:
    if args.task_count <= 0:
        raise ValueError("--task-count must be positive")
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    data_path = Path(args.data)
    model_path = Path(args.model)
    agent_config_path = Path(args.agent_config)
    reward_config_path = Path(args.reward_config)

    _require_file(data_path)
    _require_dir(model_path)
    _require_file(agent_config_path)
    _require_file(reward_config_path)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_output_dir(args)
    )
    prepare_output_dir(output_dir)

    task_pool_path = output_dir / "task_pool.jsonl"
    trajectories_path = output_dir / "trajectories.jsonl"
    groups_path = output_dir / "groups.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    complete_path = output_dir / "COMPLETE"

    started = time.perf_counter()

    manifest: dict[str, Any] = {
        "status": "running",
        "data": str(data_path),
        "model": str(model_path),
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "agent_config": str(agent_config_path),
        "reward_config": str(reward_config_path),
        "task_count": args.task_count,
        "group_size": args.group_size,
        "trajectory_count_expected": args.task_count * args.group_size,
        "selection_seed": args.selection_seed,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed_enforced_by_transformers_client": False,
        },
        "sandbox_url": os.environ.get(
            "ADAPTIVE_MATH_SANDBOX_URL",
            "http://127.0.0.1:8080",
        ),
        "parser_tolerance": list(args.parser_tolerance or []),
    }
    _write_json(manifest_path, manifest)

    try:
        frame = pd.read_parquet(data_path)
        rows = cast(
            list[dict[str, Any]],
            frame.to_dict(orient="records"),
        )

        if args.task_count > len(rows):
            raise ValueError(
                f"--task-count={args.task_count} exceeds dataset size {len(rows)}"
            )

        selected_rows = select_task_rows(
            rows,
            count=args.task_count,
            seed=args.selection_seed,
        )
        tasks = [
            row_to_labeled_task(row)
            for row in selected_rows
        ]

        with task_pool_path.open("w") as handle:
            for task in tasks:
                handle.write(task.model_dump_json() + "\n")

        budget = _load_budget(agent_config_path)
        reward_config = _load_reward(reward_config_path)

        generation = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print(f"[rollout-health] loading model: {model_path}", flush=True)

        if args.adapter is not None:
            _require_dir(args.adapter)
        model = TransformersModelClient.from_pretrained(
            str(model_path),
            device=args.device,
            dtype=args.dtype,
            adapter=str(args.adapter) if args.adapter is not None else None,
        )

        sandbox = SandboxFusionClient()

        registry = ToolRegistry(
            [
                cast(Tool, SympyTool()),
                cast(Tool, PythonTool(sandbox)),
            ]
        )

        agent = AgentLoop(parser_tolerance=args.parser_tolerance or ())

        all_rewards: list[float] = []
        reward_groups: list[list[float]] = []
        trajectory_records: list[dict[str, Any]] = []

        termination_counts: Counter[str] = Counter()

        correct_count = 0
        final_answer_count = 0
        verifier_evaluated_count = 0

        tool_calls_total = 0
        invalid_actions_total = 0
        generated_tokens_total = 0
        python_seconds_total = 0.0

        theoretical_max_generated_tokens = (
            args.max_new_tokens * budget.max_steps
        )

        try:
            for task_index, task in enumerate(tasks, start=1):
                task_id = task.task.task_id
                group_rewards: list[float] = []

                print(
                    f"[rollout-health] task "
                    f"{task_index}/{len(tasks)} "
                    f"id={task_id}",
                    flush=True,
                )

                for sample_index in range(args.group_size):
                    trace_id = (
                        f"rollout-health:{task_id}:{sample_index}"
                    )

                    environment = OfflineMathEnv(
                        task,
                        budget,
                        registry,
                        trace_id=trace_id,
                    )

                    rollout_started = time.perf_counter()

                    trajectory = await agent.run(
                        environment,
                        model,
                        generation,
                    )

                    rollout_seconds = (
                        time.perf_counter() - rollout_started
                    )

                    verdict = environment.evaluate()

                    reward = 0.0
                    reward_payload: dict[str, Any] | None = None

                    if verdict is not None:
                        verifier_evaluated_count += 1

                        breakdown = reward_for_trajectory(
                            trajectory,
                            verdict,
                            budget,
                            max_generated_tokens=(
                                theoretical_max_generated_tokens
                            ),
                            config=reward_config,
                        )

                        reward = float(breakdown.total)
                        reward_payload = breakdown.model_dump(
                            mode="json"
                        )

                        if verdict.reward > 0:
                            correct_count += 1

                    if trajectory.final_answer is not None:
                        final_answer_count += 1

                    termination = trajectory.termination_reason.value
                    termination_counts[termination] += 1

                    usage = trajectory.usage

                    tool_calls_total += usage.tool_calls
                    invalid_actions_total += usage.invalid_actions
                    generated_tokens_total += usage.generated_tokens
                    python_seconds_total += usage.python_seconds

                    group_rewards.append(reward)
                    all_rewards.append(reward)

                    record = {
                        "task_index": task_index - 1,
                        "sample_index": sample_index,
                        "task_id": task_id,
                        "trace_id": trace_id,
                        "reward": reward,
                        "rollout_seconds": rollout_seconds,
                        "trajectory": trajectory.model_dump(
                            mode="json"
                        ),
                        "verdict": (
                            verdict.model_dump(mode="json")
                            if verdict is not None
                            else None
                        ),
                        "reward_breakdown": reward_payload,
                    }

                    trajectory_records.append(record)
                    _append_jsonl(
                        trajectories_path,
                        record,
                    )

                    print(
                        f"  sample={sample_index + 1}/"
                        f"{args.group_size} "
                        f"reward={reward:.3f} "
                        f"termination={termination} "
                        f"tools={usage.tool_calls} "
                        f"tokens={usage.generated_tokens}",
                        flush=True,
                    )

                reward_groups.append(group_rewards)

                advantages = group_advantages(group_rewards)

                group_record = {
                    "task_id": task_id,
                    "rewards": group_rewards,
                    "advantages": list(advantages.advantages),
                    "effective": advantages.effective,
                    "mean_reward": advantages.mean_reward,
                    "std_reward": advantages.std_reward,
                }

                _append_jsonl(
                    groups_path,
                    group_record,
                )

                print(
                    "  group "
                    f"mean={advantages.mean_reward:.3f} "
                    f"std={advantages.std_reward:.3f} "
                    f"effective={advantages.effective}",
                    flush=True,
                )

        finally:
            await sandbox.aclose()

        trajectory_count = len(trajectory_records)

        if trajectory_count != args.task_count * args.group_size:
            raise RuntimeError(
                "trajectory count mismatch: "
                f"expected {args.task_count * args.group_size}, "
                f"got {trajectory_count}"
            )

        group_summary = summarize_group_rewards(
            reward_groups
        )

        mean_reward = statistics.fmean(all_rewards)
        reward_std = (
            statistics.pstdev(all_rewards)
            if len(all_rewards) > 1
            else 0.0
        )

        summary: dict[str, Any] = {
            **group_summary,
            "task_count": args.task_count,
            "group_size": args.group_size,
            "trajectory_count": trajectory_count,
            "mean_reward": mean_reward,
            "reward_std": reward_std,
            "correct_count": correct_count,
            "verifier_accuracy": (
                correct_count / trajectory_count
            ),
            "final_answer_count": final_answer_count,
            "valid_final_rate": (
                final_answer_count / trajectory_count
            ),
            "verifier_evaluated_count": (
                verifier_evaluated_count
            ),
            "verifier_evaluated_rate": (
                verifier_evaluated_count / trajectory_count
            ),
            "termination_counts": dict(
                sorted(termination_counts.items())
            ),
            "tool_calls_total": tool_calls_total,
            "tool_calls_mean": (
                tool_calls_total / trajectory_count
            ),
            "invalid_actions_total": (
                invalid_actions_total
            ),
            "invalid_actions_mean": (
                invalid_actions_total / trajectory_count
            ),
            "generated_tokens_total": (
                generated_tokens_total
            ),
            "generated_tokens_mean": (
                generated_tokens_total / trajectory_count
            ),
            "python_seconds_total": (
                python_seconds_total
            ),
            "python_seconds_mean": (
                python_seconds_total / trajectory_count
            ),
            "elapsed_seconds": (
                time.perf_counter() - started
            ),
        }

        _write_json(summary_path, summary)

        manifest["status"] = "complete"
        manifest["summary"] = summary
        _write_json(manifest_path, manifest)

        complete_path.write_text("complete\n")

        print(
            "\n[rollout-health] COMPLETE",
            flush=True,
        )
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

        return output_dir

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["elapsed_seconds"] = (
            time.perf_counter() - started
        )
        _write_json(manifest_path, manifest)
        raise


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
