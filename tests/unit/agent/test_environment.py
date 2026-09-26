import asyncio

from adaptive_math.agent.actions import FinalAction, ToolAction, ToolCall
from adaptive_math.agent.environment import OfflineMathEnv, ProductMathEnv
from adaptive_math.agent.state import TerminationReason
from adaptive_math.core.types import AnswerType, Budget, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.tools.base import ToolContext, ToolResult
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.verifier.service import VerifierStatus


class EchoArguments:
    @classmethod
    def model_validate(cls, value: object) -> object:
        return value

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        return {}


class EchoTool:
    name = "echo"
    description = "echo"
    arguments_model = EchoArguments

    async def execute(self, arguments: object, context: ToolContext) -> ToolResult:
        return ToolResult(ok=True, output="observed", latency_ms=1)


def _task() -> MathTask:
    return MathTask(
        task_id="unit:env",
        problem="What is 1 + 1?",
        answer_type=AnswerType.INTEGER,
        dataset="unit",
        split="train",
        source_hash="a" * 64,
        pipeline_version="unit-v1",
    )


def _budget() -> Budget:
    return Budget(max_steps=3, max_tool_calls=1, max_python_seconds=1.0, max_observation_chars=100)


def test_product_env_records_tool_observation_without_reference() -> None:
    environment = ProductMathEnv(_task(), _budget(), ToolRegistry([EchoTool()]))

    result = asyncio.run(
        environment.step(ToolAction(call=ToolCall(name="echo", arguments={})), monotonic_ms=1)
    )

    assert not result.terminated
    assert result.observation is not None and result.observation.content == "observed"
    assert result.state.usage.steps == 1
    assert result.state.usage.tool_calls == 1
    assert "reference" not in result.state.model_dump_json().lower()


def test_final_terminates_and_offline_evaluation_is_separate() -> None:
    labeled = LabeledMathTask(
        task=_task(), reference=ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER)
    )
    environment = OfflineMathEnv(labeled, _budget(), ToolRegistry([]))

    result = asyncio.run(environment.step(FinalAction(answer="2"), monotonic_ms=2))
    evaluation = environment.evaluate()

    assert result.terminated
    assert result.state.termination_reason is TerminationReason.FINAL
    assert evaluation is not None and evaluation.reward == 1.0
    assert "reference" not in result.state.model_dump_json().lower()


class ContextCapturingPythonTool:
    name = "python"
    description = "records the budget it was given"
    arguments_model = EchoArguments

    def __init__(self, execution_time: float) -> None:
        self._execution_time = execution_time
        self.contexts: list[ToolContext] = []

    async def execute(self, arguments: object, context: ToolContext) -> ToolResult:
        self.contexts.append(context)
        return ToolResult(
            ok=True, output="done", latency_ms=1, metadata={"execution_time": self._execution_time}
        )


def test_remaining_python_budget_shrinks_with_charged_sandbox_time() -> None:
    """The tool sees what is left, not what was granted: otherwise a second slow
    run can overrun the episode budget the reward is normalized against."""
    budget = Budget(
        max_steps=4, max_tool_calls=4, max_python_seconds=10.0, max_observation_chars=100
    )
    tool = ContextCapturingPythonTool(execution_time=3.0)
    environment = ProductMathEnv(_task(), budget, ToolRegistry([tool]))
    call = ToolAction(call=ToolCall(name="python", arguments={}))

    asyncio.run(environment.step(call, monotonic_ms=1))
    asyncio.run(environment.step(call, monotonic_ms=2))

    assert [context.remaining_python_seconds for context in tool.contexts] == [10.0, 7.0]


def test_tool_budget_is_reserved_and_exhaustion_still_allows_final() -> None:
    environment = ProductMathEnv(_task(), _budget(), ToolRegistry([EchoTool()]))
    first = asyncio.run(environment.step(ToolAction(call=ToolCall(name="echo", arguments={})), 1))
    second = asyncio.run(environment.step(ToolAction(call=ToolCall(name="echo", arguments={})), 2))
    final = asyncio.run(environment.step(FinalAction(answer="2"), 3))

    assert first.state.usage.tool_calls == 1
    assert second.observation is not None and second.observation.kind == "action_error"
    assert final.state.termination_reason is TerminationReason.FINAL


def _labeled() -> LabeledMathTask:
    return LabeledMathTask(task=_task(), reference=ReferenceAnswer(value="2", answer_type=AnswerType.INTEGER))


def _exhaust_budget(environment: OfflineMathEnv, steps: int) -> None:
    for monotonic_ms in range(1, steps + 1):
        asyncio.run(environment.step(None, monotonic_ms=monotonic_ms))


def test_budget_exhaustion_without_a_final_answer_is_an_invalid_prediction() -> None:
    """The eval loop scores "no extractable answer" as INVALID_PREDICTION. The
    training loop scored the same fact as *no verdict at all*, which skipped
    reward_for_trajectory entirely: budget exhaustion paid 0.0 while an honest
    wrong answer carrying invalid actions paid -0.2, so giving up outranked
    terminating. Both loops must judge one fact the same way."""
    environment = OfflineMathEnv(_labeled(), _budget(), ToolRegistry([]))
    _exhaust_budget(environment, _budget().max_steps)

    verdict = environment.evaluate()

    assert environment.state.termination_reason is TerminationReason.MAX_STEPS
    assert verdict is not None
    assert verdict.status is VerifierStatus.INVALID_PREDICTION
    assert verdict.reward == 0.0
    assert verdict.details["error_code"] == "no_final_answer"


def test_an_unfinished_rollout_still_has_no_verdict() -> None:
    """Only a terminated rollout may be scored; an in-flight one must not be
    reported as an invalid prediction."""
    environment = OfflineMathEnv(_labeled(), _budget(), ToolRegistry([]))
    asyncio.run(environment.step(None, monotonic_ms=1))

    assert environment.evaluate() is None


def test_the_no_answer_verdict_carries_no_reference_text() -> None:
    """The hidden-verifier boundary is that labels never ride out through the
    public verdict path -- including the synthetic one."""
    environment = OfflineMathEnv(_labeled(), _budget(), ToolRegistry([]))
    _exhaust_budget(environment, _budget().max_steps)

    verdict = environment.evaluate()

    assert verdict is not None
    assert verdict.normalized_prediction is None
    assert "2" not in str(verdict.details)
