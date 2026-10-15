"""Section 5 gate item: does a recorded generation seed really reproduce?

Rebuilds the first recorded Stage A turn -- same task, same prompt, same
declared seed -- and compares the fresh completion with the text the run
recorded, byte for byte. A second call under the *same* seed proves the call is
deterministic; a third under the *next* seed in the scheme shows the text moves
when the seed moves, which is what "the seed is effective" has to mean.

Usage mirrors the stage runner, because the comparison is only meaningful when
the model, agent config, parser tolerance and sampling parameters are the ones
the recorded run used.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent


def _load_runner() -> ModuleType:
    """Import run_feedback_recovery.py by path (the scripts directory is not a
    package) so the check reuses the exact model/context construction."""
    spec = importlib.util.spec_from_file_location(
        "run_feedback_recovery", HERE / "run_feedback_recovery.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _first_recorded_turn(run_dir: Path) -> dict[str, Any]:
    """The first recorded trajectory that actually called a tool, with the
    first model turn kept separately for the single-turn comparison."""
    path = run_dir / "stage_a" / "trajectories.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        trajectory = record["trajectory"]
        if trajectory["usage"]["tool_calls"] < 1:
            continue
        first = next(
            event for event in trajectory["events"] if event["kind"] == "model_output"
        )
        return {
            "task_id": record["task_id"],
            "task_index": record["task_index"],
            "sample_index": record["sample_index"],
            "raw": str(first["payload"]["raw"]),
            "generated_tokens": first["payload"]["generated_tokens"],
            "events": trajectory["events"],
            "usage": trajectory["usage"],
        }
    raise SystemExit(f"no recorded model turn with an executed tool call in {path}")


def _without_wall_clock(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the fields that measure how long the machine took, keep everything
    the model saw or produced. ``monotonic_ms``, the sandbox's ``latency_ms``
    and any execution timing are the only things that legitimately differ
    between two runs of the same seeded episode."""
    timing_keys = {"execution_time", "wall_time", "duration", "latency_ms"}
    stripped: list[dict[str, Any]] = []
    for event in events:
        copy = {key: value for key, value in event.items() if key != "monotonic_ms"}
        if copy.get("kind") == "tool_result":
            payload = dict(copy["payload"])
            result = dict(payload.get("result", {}))
            result = {key: value for key, value in result.items() if key not in timing_keys}
            result["metadata"] = {
                key: value
                for key, value in dict(result.get("metadata") or {}).items()
                if key not in timing_keys
            }
            payload["result"] = result
            copy["payload"] = payload
        stripped.append(copy)
    return stripped


async def _replay_episode(
    args: argparse.Namespace,
    runner: ModuleType,
    context: object,
    task: object,
    recorded: dict[str, Any],
) -> dict[str, Any]:
    """Re-run the whole recorded episode under its own seed schedule.

    Turn 0 matching is necessary but not sufficient: the claim the audit has to
    support is that the *conversation* is reproduced, observations included. So
    the full event sequence is compared, minus wall-clock fields."""
    sample_index = int(recorded["sample_index"])
    fresh = await runner._roll_one(context, task, stage="a", sample_index=sample_index)
    fresh_record = runner._rollout_record(
        fresh, stage="a", index=int(recorded["task_index"]), args=args
    )
    before = _without_wall_clock(recorded["events"])
    after = _without_wall_clock(fresh_record["trajectory"]["events"])
    before_usage = recorded["usage"]
    after_usage = fresh_record["trajectory"]["usage"]
    return {
        "episode_events_identical": before == after,
        "origin_events": len(before),
        "replay_events": len(after),
        "first_difference": _first_difference(before, after),
        "usage_identical_except_python_seconds": {
            key: value for key, value in before_usage.items() if key != "python_seconds"
        }
        == {key: value for key, value in after_usage.items() if key != "python_seconds"},
        "origin_python_seconds": before_usage.get("python_seconds"),
        "replay_python_seconds": after_usage.get("python_seconds"),
        "replay_correct": fresh_record["correct"],
    }


def _first_difference(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> str | None:
    for index, (left, right) in enumerate(zip(before, after, strict=False)):
        if left != right:
            return f"event {index}: {json.dumps(left, ensure_ascii=False)[:300]} != {json.dumps(right, ensure_ascii=False)[:300]}"
    if len(before) != len(after):
        return f"event count {len(before)} != {len(after)}"
    return None


async def _run(args: argparse.Namespace, runner: ModuleType) -> int:
    from adaptive_math.agent.prompts import render_initial_messages
    from adaptive_math.recovery.seeds import generation_seed

    run_dir = Path(args.run_dir)
    recorded = _first_recorded_turn(run_dir)
    context = runner._build_context(args)
    tasks = {task.task.task_id: task for task in runner._load_tasks(Path(args.data))}
    task = tasks[str(recorded["task_id"])]
    messages = tuple(render_initial_messages(task.task, context.budget, context.registry))

    def seed_for(sample_index: int) -> int:
        return generation_seed(
            stage="a",
            task_id=str(recorded["task_id"]),
            sample_id=f"{recorded['task_id']}::s{sample_index}",
            arm="none",
            turn=0,
        )

    same_seed = seed_for(int(recorded["sample_index"]))
    other_seed = seed_for(int(recorded["sample_index"]) + 1)
    texts: list[str] = []
    for seed in (same_seed, same_seed, other_seed):
        turn = await context.model.generate(
            messages, context.generation.model_copy(update={"seed": seed})
        )
        texts.append(turn.text)
        print(f"[seed-check] seed={seed} tokens={turn.generated_tokens} chars={len(turn.text)}")

    recorded_raw = str(recorded["raw"])
    replay_matches = texts[0] == recorded_raw and texts[1] == recorded_raw
    seed_moves_text = texts[0] != texts[2]

    print("[seed-check] replaying the full episode ...")
    episode = await _replay_episode(args, runner, context, task, recorded)
    print(f"[seed-check] episode identical={episode['episode_events_identical']}")

    payload = {
        "task_id": recorded["task_id"],
        "sample_index": recorded["sample_index"],
        "same_seed": same_seed,
        "other_seed": other_seed,
        "recorded_chars": len(recorded_raw),
        "replay_chars": [len(text) for text in texts],
        "replay_matches_recorded": replay_matches,
        "replay_is_deterministic": texts[0] == texts[1],
        "different_seed_changes_text": seed_moves_text,
        **episode,
        "pass": bool(
            replay_matches
            and seed_moves_text
            and episode["episode_events_identical"]
            and episode["usage_identical_except_python_seconds"]
        ),
        "model_version": context.model_version,
    }
    (run_dir / "stage_a" / "seed_check.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["pass"] else 1


def main() -> int:
    runner = _load_runner()
    parser = runner.build_parser()
    parser.description = __doc__
    args = parser.parse_args()
    # The runner's own _main does this after parsing; mirror it so the agent
    # gets the same tolerance tuple and the replay is a real replay.
    args.parser_tolerance = runner._parse_tolerance(args.parser_tolerance)
    return asyncio.run(_run(args, runner))


if __name__ == "__main__":
    raise SystemExit(main())
