"""E1: offline counterfactual for parser-tolerance harness patches.

Population (pre-registration addendum A): agent-path trajectories only. The
direct-eval arms score with verifier.extract and never call parse_action, so a
parser patch cannot change them.

Two single-field candidate rules, each evaluated separately:
  A  strip a leading think block that is never closed, then re-parse
  B  accept a bare scalar inside the final tag (no JSON object)

Trajectory-level counterfactuals are computed only where they are identifiable
offline: the first model turn of a trajectory fails to parse, a rule rescues it
into a FinalAction, and the counterfactual rollout therefore terminates on that
turn. Verdicts are recomputed with the production verifier against references
reproduced from the same task pool (task id, seed and selection code), and
rewards are recomputed with compute_reward and the run's own reward config.
Anything else (a rescue mid-trajectory, or a rescued ToolAction whose
observation would change later turns) is counted as NOT_IDENTIFIABLE and must
be settled by a paired re-run, not by extrapolation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_math.agent.parser import parse_action
from adaptive_math.core.types import Budget, LabeledMathTask, ReferenceAnswer
from adaptive_math.reward import RewardConfig, compute_reward
from adaptive_math.reward.types import RewardContext
from adaptive_math.verifier import verify_answer

THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"
FINAL_CLOSE = "</" + "final>"
FINAL_OPEN = "<" + "final>"


def rule_a(raw: str) -> str:
    """Drop a leading think block that the model never closed."""
    text = raw.strip()
    if not text.startswith(THINK_OPEN) or THINK_CLOSE in text:
        return raw
    index = text.find(FINAL_OPEN)
    tool_index = text.find("<" + "tool_call>")
    if tool_index != -1 and (index == -1 or tool_index < index):
        index = tool_index
    return text[index:] if index != -1 else raw


def rule_b(raw: str) -> str:
    """Wrap a bare scalar final payload so it parses as a JSON answer object."""
    text = raw.strip()
    match = re.search(re.escape(FINAL_OPEN) + r"\s*([^<{}]+?)\s*" + re.escape(FINAL_CLOSE), text, re.DOTALL)
    if match is None:
        return raw
    payload = match.group(1).strip()
    if payload.startswith("{"):
        return raw
    replacement = FINAL_OPEN + json.dumps({"answer": payload}, ensure_ascii=False) + FINAL_CLOSE
    return text[: match.start()] + replacement + text[match.end() :]


RULES = {"A_unclosed_think": rule_a, "B_bare_final_scalar": rule_b}


def _references(run_dir: Path) -> dict[str, ReferenceAnswer]:
    pool_path = run_dir / "task_pool.jsonl"
    references: dict[str, ReferenceAnswer] = {}
    for line in pool_path.read_text().splitlines():
        if line.strip():
            labeled = LabeledMathTask.model_validate_json(line)
            references[labeled.task.task_id] = labeled.reference
    return references


def _budget(agent_config: Path) -> Budget:
    return Budget.model_validate(yaml.safe_load(agent_config.read_text()))


def _reward_config(path: Path) -> RewardConfig:
    return RewardConfig.model_validate(yaml.safe_load(path.read_text()))


def _reward(
    *,
    verdict,
    invalid_actions: int,
    generated_tokens: int,
    max_generated_tokens: int,
    budget: Budget,
    config: RewardConfig,
) -> float:
    """Recompute a reward with the production implementation, from a real verdict."""
    context = RewardContext(
        verifier_result=verdict,
        tool_calls=0,
        python_seconds=0.0,
        invalid_action_count=invalid_actions,
        generated_tokens=generated_tokens,
        max_generated_tokens=max_generated_tokens,
        budget=budget,
    )
    return compute_reward(context, config).total


def analyze_run(run_dir: Path, *, max_generated_tokens: int) -> dict:
    trajectories = [
        json.loads(line)
        for line in (run_dir / "trajectories.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    references = _references(run_dir)
    budget = _budget(Path(manifest["agent_config"]))
    config = _reward_config(Path(manifest["reward_config"]))

    turn_transitions: dict[str, Counter] = {name: Counter() for name in RULES}
    turn_shape = Counter()
    counterfactuals: dict[str, list[dict]] = {name: [] for name in RULES}
    not_identifiable: dict[str, int] = {name: 0 for name in RULES}

    for record in trajectories:
        trajectory = record["trajectory"]
        task_id = record["task_id"]
        reference = references[task_id]
        model_turns = [e for e in trajectory["events"] if e["kind"] == "model_output"]
        actual_reward = float(record["reward"])
        actual_correct = (record.get("verdict") or {}).get("status") == "correct"

        # Per rule, the first turn whose parse outcome changes decides the
        # counterfactual: a rescued FinalAction ends the rollout there, so later
        # turns never happen and the outcome is exactly computable offline. A
        # rescued ToolAction would execute and rewrite the context, so that
        # trajectory is not identifiable without a paired re-run.
        decided: dict[str, bool] = {name: False for name in RULES}

        for position, event in enumerate(model_turns):
            raw = event["payload"]["raw"]
            current = parse_action(raw)
            current_error = str(current.error) if current.error else "ok"
            stripped = raw.strip()
            if stripped.startswith(THINK_OPEN) and THINK_CLOSE not in stripped:
                turn_shape["unclosed_think"] += 1
            if FINAL_OPEN in stripped and FINAL_CLOSE in stripped:
                turn_shape["has_final_tags"] += 1
            for name, rule in RULES.items():
                candidate = parse_action(rule(raw))
                candidate_error = str(candidate.error) if candidate.error else "ok"
                turn_transitions[name][f"{current_error}->{candidate_error}"] += 1
                if decided[name] or current_error == "ok" or candidate_error != "ok":
                    # Either already resolved, or this turn behaves identically
                    # under the patch (still accepted, or still rejected with an
                    # invalid-action notice), so the real rollout is unchanged.
                    continue
                kind = candidate.action.kind if candidate.action is not None else None
                if kind != "final":
                    not_identifiable[name] += 1
                    decided[name] = True
                    continue
                tokens_so_far = sum(
                    int(e["payload"].get("generated_tokens", 0))
                    for e in model_turns[: position + 1]
                )
                invalid_before = sum(
                    1
                    for e in model_turns[:position]
                    if parse_action(e["payload"]["raw"]).error is not None
                )
                answer = candidate.action.answer
                verdict = verify_answer(answer, reference, task_id=task_id)
                correct = verdict.status.value == "correct"
                cf_reward = _reward(
                    verdict=verdict,
                    invalid_actions=invalid_before,
                    generated_tokens=tokens_so_far,
                    max_generated_tokens=max_generated_tokens,
                    budget=budget,
                    config=config,
                )
                counterfactuals[name].append(
                    {
                        "task_id": task_id,
                        "task_index": record["task_index"],
                        "sample_index": record["sample_index"],
                        "rescued_at_turn": position,
                        "answer": answer,
                        "actual_termination": str(trajectory.get("termination_reason")),
                        "counterfactual_termination": "final",
                        "actual_reward": actual_reward,
                        "counterfactual_reward": round(cf_reward, 4),
                        "actual_correct": actual_correct,
                        "counterfactual_correct": correct,
                        "verifier_status": verdict.status.value,
                        "tokens": tokens_so_far,
                        "invalid_actions_before_rescue": invalid_before,
                    }
                )
                decided[name] = True

    gates: dict[str, dict] = {}
    for name, rows in counterfactuals.items():
        gained = [r for r in rows if r["counterfactual_correct"] and not r["actual_correct"]]
        demoted = [r for r in rows if r["actual_correct"] and not r["counterfactual_correct"]]
        both = [r for r in rows if r["actual_correct"] and r["counterfactual_correct"]]
        neither = [r for r in rows if not r["actual_correct"] and not r["counterfactual_correct"]]
        reward_delta = sum(r["counterfactual_reward"] - r["actual_reward"] for r in rows)
        gates[name] = {
            "identifiable_trajectories": len(rows),
            "not_identifiable_turns": not_identifiable[name],
            "correct_gained": len(gained),
            "correct_demoted": len(demoted),
            "correct_both": len(both),
            "correct_neither": len(neither),
            "reward_delta_total": round(reward_delta, 4),
            "gate_1_gain_at_least_one": len(gained) >= 1,
            "gate_2_zero_demotions": len(demoted) == 0,
            "transition_table": dict(turn_transitions[name]),
            "rows": rows,
        }

    return {
        "run_dir": str(run_dir),
        "trajectory_count": len(trajectories),
        "turn_shape": dict(turn_shape),
        "rules": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--max-generated-tokens", type=int, default=6144,
                        help="budget.max_steps * generation cap, as the run used it")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    reports = [analyze_run(d, max_generated_tokens=args.max_generated_tokens) for d in args.run_dir]
    text = json.dumps(reports, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
