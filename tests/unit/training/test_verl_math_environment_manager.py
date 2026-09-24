import asyncio
from types import SimpleNamespace

import numpy as np

from adaptive_math.core.types import AnswerType, Budget, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.reward import RewardConfig
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.training.verl_environment import MathRolloutManager, VerlMathEnvironmentManager


def _task(task_id: str = "unit:verl") -> LabeledMathTask:
    return LabeledMathTask(
        task=MathTask(
            task_id=task_id,
            problem="1+1",
            answer_type=AnswerType.INTEGER,
            dataset="unit",
            split="train",
            source_hash="a" * 64,
            pipeline_version="v1",
        ),
        reference=ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER),
    )


def test_verl_manager_uses_production_env_for_terminal_reward() -> None:
    config = SimpleNamespace(env=SimpleNamespace(rollout=SimpleNamespace(n=1)))
    manager = VerlMathEnvironmentManager(
        [_task()],
        Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100),
        ToolRegistry([]),
        config,
        policy_version="p1",
    )

    initial, infos = manager.reset({})
    next_observations, rewards, dones, step_infos = manager.step(['<final>{"answer":"2"}</final>'])

    assert len(initial["text"]) == 1 and "1+1" in initial["text"][0]
    assert infos == [{}]
    # Fixed-width contract: the terminated row keeps repeating its final
    # context instead of disappearing from the batch.
    assert len(next_observations["text"]) == 1 and "1+1" in next_observations["text"][0]
    assert rewards.tolist() == [1.0]
    assert dones.tolist() == [True]
    assert step_infos[0]["policy_version"] == "p1"
    assert step_infos[0]["won"] == 1.0
    assert manager.success_evaluator(
        total_batch_list=[[{"active_masks": True}]], total_infos=[[step_infos[0]]]
    )["success_rate"].tolist() == [1.0]


def test_verl_manager_rejects_non_grouped_config() -> None:
    config = SimpleNamespace(env=SimpleNamespace(rollout=SimpleNamespace(n=0)))
    try:
        VerlMathEnvironmentManager(
            [_task()],
            Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100),
            ToolRegistry([]),
            config,
            policy_version="p1",
        )
    except ValueError as exc:
        assert "group_size" in str(exc)
    else:
        raise AssertionError("expected a non-grouped rollout config to be rejected")


def test_math_rollout_manager_is_fixed_width_across_termination() -> None:
    manager = MathRolloutManager(
        Budget(max_steps=4, max_tool_calls=2, max_python_seconds=0, max_observation_chars=500),
        ToolRegistry([]),
    )
    manager.reset([_task("unit:verl:a"), _task("unit:verl:b")], group_size=1, policy_version="p1")

    first = asyncio.run(manager.step(['<final>{"answer":"2"}</final>', '<final>{"answer":"99"}</final>']))
    assert [transition.reward for transition in first] == [1.0, 0.0]
    assert [transition.done for transition in first] == [True, True]

    # Fixed-width contract: every row still accepts an action after
    # termination; terminated rows are no-ops that repeat their terminal
    # info and a zero reward.
    second = asyncio.run(manager.step(['<final>{"answer":"2"}</final>', '<final>{"answer":"99"}</final>']))
    assert [transition.reward for transition in second] == [0.0, 0.0]
    assert [transition.done for transition in second] == [True, True]
    assert second[0].info["won"] == 1.0
    assert second[1].info["won"] == 0.0
    assert len(manager.build_text_obs()) == 2


def test_math_rollout_manager_obs_carries_tool_results() -> None:
    from adaptive_math.tools.sympy_tool import SympyTool

    manager = MathRolloutManager(
        Budget(max_steps=4, max_tool_calls=2, max_python_seconds=0, max_observation_chars=500),
        ToolRegistry([SympyTool()]),
    )
    manager.reset([_task()], group_size=1, policy_version="p1")

    transition = asyncio.run(
        manager.step(['<tool_call>{"name": "sympy", "arguments": {"operation": "numeric", "expression": "1+1"}}</tool_call>'])
    )[0]

    assert transition.done is False
    observations = manager.build_text_obs()
    assert len(observations) == 1
    assert "2.00000000000000" in observations[0]
    assert "Assistant:" in observations[0] and "Observation:" in observations[0]


def test_verl_manager_reset_creates_one_env_per_row() -> None:
    config = SimpleNamespace(env=SimpleNamespace(rollout=SimpleNamespace(n=4)))
    manager = VerlMathEnvironmentManager(
        [],
        Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100),
        ToolRegistry([]),
        config,
        policy_version="p1",
        task_lookup={"unit:verl": _task()},
    )

    rows = np.array(
        [{"task_ids": ["unit:verl", "unit:verl", "unit:verl", "unit:verl"]} for _ in range(8)],
        dtype=object,
    )
    observations, infos = manager.reset(rows)

    assert len(observations["text"]) == 8
    assert len(infos) == 8


def test_verl_manager_marks_parse_failures_as_invalid_actions() -> None:
    config = SimpleNamespace(env=SimpleNamespace(rollout=SimpleNamespace(n=1)))
    manager = VerlMathEnvironmentManager(
        [_task()],
        Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100),
        ToolRegistry([]),
        config,
        policy_version="p1",
    )
    manager.reset({})

    _, rewards, dones, step_infos = manager.step(["garbage that is not an action"])

    assert rewards.tolist() == [0.0]
    assert dones.tolist() == [False]
    assert step_infos[0]["is_action_valid"] is False


def test_math_rollout_uses_versioned_reward_breakdown_at_terminal() -> None:
    reward = RewardConfig(
        version="r2-test",
        variant="r2",
        tool_weight=0.15,
        python_weight=0.10,
        invalid_weight=0.10,
        invalid_cap=3,
        token_weight=0.0,
        clip_min=-1.0,
        clip_max=1.0,
    )
    manager = MathRolloutManager(
        Budget(max_steps=2, max_tool_calls=1, max_python_seconds=1, max_observation_chars=100),
        ToolRegistry([]),
        reward_config=reward,
    )
    manager.reset([_task()], group_size=1, policy_version="p1")

    transition = asyncio.run(manager.step(['<final>{"answer":"2"}</final>']))[0]

    assert transition.reward == 1.0
    assert transition.info["reward_version"] == "r2-test"
    assert transition.info["reward_components"] == {
        "correct": 1.0,
        "invalid_penalty": 0.0,
        "python_cost": 0.0,
        "token_cost": 0.0,
        "tool_cost": 0.0,
    }


def test_budget_exhaustion_does_not_outscore_an_honest_wrong_answer() -> None:
    """The reward shape GRPO actually saw during R2: a row that burned its budget
    on invalid actions scored 0.0, while a row that admitted a wrong answer scored
    -0.1. Within one group that ranks giving up above trying, and it is why the
    invalid-action penalty never reaches the trajectories that need it most."""
    import pytest

    reward = RewardConfig(
        version="r2-test",
        variant="r2",
        tool_weight=0.15,
        python_weight=0.10,
        invalid_weight=0.10,
        invalid_cap=3,
        token_weight=0.0,
        clip_min=-1.0,
        clip_max=1.0,
    )
    manager = MathRolloutManager(
        Budget(max_steps=2, max_tool_calls=1, max_python_seconds=1, max_observation_chars=100),
        ToolRegistry([]),
        reward_config=reward,
    )
    manager.reset(
        [_task("unit:exhaust"), _task("unit:honest")], group_size=1, policy_version="p1"
    )

    asyncio.run(manager.step(["not an action", "not an action"]))
    final = asyncio.run(
        manager.step(["not an action", '<final>{"answer":"99"}</final>'])
    )

    exhausted, honest = final[0], final[1]
    assert exhausted.info["termination_reason"] == "max_steps"
    assert honest.info["termination_reason"] == "final"
    assert exhausted.reward == pytest.approx(-0.2)
    assert honest.reward == pytest.approx(-0.1)
    assert exhausted.reward < honest.reward
