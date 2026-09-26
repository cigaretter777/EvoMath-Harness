"""AgentLoop 接入 HarnessSpec：溯源字段与运行参数一致性校验。"""

import asyncio

import pytest

from adaptive_math.agent.environment import ProductMathEnv
from adaptive_math.agent.loop import RUNTIME_VERSION, AgentLoop
from adaptive_math.agent.model_client import GenerationConfig, ModelTurn
from adaptive_math.agent.prompts import PROMPT_VERSION
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import AnswerType, Budget, MathTask
from adaptive_math.harness.spec import HarnessSpec
from adaptive_math.reward.functions import R2_DEFAULT
from adaptive_math.tools.registry import ToolRegistry

_BUDGET = Budget(max_steps=3, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100)


def _spec(**overrides: object) -> HarnessSpec:
    fields: dict[str, object] = {
        "harness_version": "h-test",
        "prompt_version": PROMPT_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "tools": (),
        "budget": _BUDGET,
        "generation": GenerationConfig(),
        "reward": R2_DEFAULT,
    }
    fields.update(overrides)
    return HarnessSpec(**fields)  # type: ignore[arg-type]


class ScriptedModel:
    def __init__(self, turns: list[str]) -> None:
        self._turns = iter(turns)

    async def generate(self, messages: tuple[object, ...], config: GenerationConfig) -> ModelTurn:
        return ModelTurn(
            text=next(self._turns),
            prompt_tokens=1,
            generated_tokens=2,
            finish_reason="stop",
            model_id="scripted",
        )


def _environment(budget: Budget = _BUDGET, registry: ToolRegistry | None = None) -> ProductMathEnv:
    task = MathTask(
        task_id="unit:harness-loop",
        problem="What is 1 + 1?",
        answer_type=AnswerType.INTEGER,
        dataset="unit",
        split="test",
        source_hash="a" * 64,
        pipeline_version="unit-v1",
    )
    return ProductMathEnv(task, budget, registry if registry is not None else ToolRegistry([]))


def _run(loop: AgentLoop, environment: ProductMathEnv, generation: GenerationConfig) -> object:
    return asyncio.run(
        loop.run(environment, ScriptedModel(['<final>{"answer":"2"}</final>']), generation)
    )


def test_harness_loop_stamps_runtime_tag_and_spec_hash() -> None:
    spec = _spec()
    trajectory = _run(AgentLoop(harness=spec), _environment(), GenerationConfig())

    assert trajectory.runtime_version == spec.runtime_tag() == "runtime-v1+agent-v1+h-test"
    assert trajectory.harness_spec_hash == spec.spec_hash()


def test_legacy_loop_keeps_two_part_runtime_version_and_no_spec_hash() -> None:
    trajectory = _run(AgentLoop(), _environment(), GenerationConfig())

    assert trajectory.runtime_version == f"{RUNTIME_VERSION}+{PROMPT_VERSION}"
    assert trajectory.harness_spec_hash is None
    assert trajectory.model_version is None


def test_loop_stamps_model_version_when_configured() -> None:
    trajectory = _run(
        AgentLoop(harness=_spec(), model_version="ckpt-qwen3-sft-v1"),
        _environment(),
        GenerationConfig(),
    )

    assert trajectory.model_version == "ckpt-qwen3-sft-v1"


def test_harness_loop_rejects_generation_mismatch() -> None:
    with pytest.raises(ValueError, match="generation"):
        _run(
            AgentLoop(harness=_spec()),
            _environment(),
            GenerationConfig(max_new_tokens=64),
        )


def test_harness_loop_rejects_budget_mismatch() -> None:
    other = Budget(max_steps=2, max_tool_calls=0, max_python_seconds=0, max_observation_chars=100)
    with pytest.raises(ValueError, match="budget"):
        _run(AgentLoop(harness=_spec()), _environment(budget=other), GenerationConfig())


def test_harness_loop_rejects_tool_set_mismatch() -> None:
    class _FakeTool:
        name = "python"
        description = "fake"
        arguments_model = object()

    with pytest.raises(ValueError, match="tool set"):
        _run(
            AgentLoop(harness=_spec()),
            _environment(registry=ToolRegistry([_FakeTool()])),
            GenerationConfig(),
        )


def test_harness_and_legacy_loops_produce_identical_episode_behaviour() -> None:
    """Phase H1 门禁：同一剧本下两种构造的事件流逐事件一致（monotonic_ms 为
    挂钟噪声，不参与比较）；差异仅允许在溯源字段。"""
    script = ["not protocol", '<tool>{"name":"python"}</tool>', '<final>{"answer":"2"}</final>']
    legacy = asyncio.run(
        AgentLoop().run(_environment(), ScriptedModel(list(script)), GenerationConfig())
    )
    harnessed = asyncio.run(
        AgentLoop(harness=_spec()).run(_environment(), ScriptedModel(list(script)), GenerationConfig())
    )

    def signature(trajectory: Trajectory) -> object:
        events = [(event.sequence, event.kind, event.payload) for event in trajectory.events]
        return (events, trajectory.usage, trajectory.final_answer, trajectory.termination_reason)

    assert signature(harnessed) == signature(legacy)
    assert harnessed.runtime_version != legacy.runtime_version
    assert harnessed.harness_spec_hash is not None
