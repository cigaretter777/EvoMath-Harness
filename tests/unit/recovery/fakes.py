"""Deterministic model and tool doubles for the recovery tests.

The model is a pure function of the conversation it is handed (it picks its
action by counting assistant turns), so a prefix resume continues the exact
script the uninterrupted run would have followed. The tool fails on demand
with a real traceback-shaped payload, which is what the environment turns into
the E_EXEC observation the anchors are built on.
"""

from typing import Any

from pydantic import BaseModel

from adaptive_math.agent.model_client import ChatMessage, GenerationConfig, ModelTurn
from adaptive_math.tools.base import ToolContext, ToolErrorCode, ToolResult


class LooseArguments(BaseModel):
    """Accepts any object: the doubles never validate call shape."""

    model_config = {"extra": "allow"}

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        return {}


class ScriptedModel:
    def __init__(self, script: list[str], *, model_id: str = "scripted") -> None:
        self.script = script
        self.model_id = model_id
        self.configs: list[GenerationConfig] = []
        self.prompts: list[tuple[ChatMessage, ...]] = []

    @property
    def turn(self) -> int:
        return len(self.configs)

    async def generate(
        self, messages: tuple[ChatMessage, ...], config: GenerationConfig
    ) -> ModelTurn:
        turn = sum(1 for message in messages if message.role == "assistant")
        text = self.script[turn] if turn < len(self.script) else '<final>{"answer":"0"}</final>'
        self.configs.append(config)
        self.prompts.append(messages)
        return ModelTurn(
            text=text,
            prompt_tokens=len(messages),
            generated_tokens=7,
            finish_reason="stop",
            model_id=self.model_id,
        )


class FakePythonTool:
    """Fails on code containing ``BOOM``; otherwise prints and returns 0.25 s."""

    name = "python"
    description = "fake python"
    arguments_model = LooseArguments

    def __init__(self, *, fail_times: int = 0, execution_time: float = 0.25) -> None:
        self.fail_times = fail_times
        self.execution_time = execution_time
        self.calls: list[str] = []

    async def execute(self, arguments: Any, context: ToolContext) -> ToolResult:
        code = str(getattr(arguments, "code", "") or arguments)
        self.calls.append(code)
        if "BOOM" in code:
            return ToolResult(
                ok=False,
                output=(
                    "stdout:\n\nstderr:\nTraceback (most recent call last):\n"
                    '  File "<stdin>", line 1, in <module>\n'
                    "ZeroDivisionError: division by zero\n"
                ),
                error_code=ToolErrorCode.EXECUTION_ERROR,
                latency_ms=4,
                metadata={
                    "execution_time": self.execution_time,
                    "sandbox_status": "Finished",
                    "return_code": 1,
                },
            )
        return ToolResult(
            ok=True,
            output="stdout:\n42\n",
            latency_ms=3,
            metadata={"execution_time": self.execution_time},
        )


class DeadSandboxTool(FakePythonTool):
    """Every call reports an unreachable service (E_INFRA, never the model's fault)."""

    async def execute(self, arguments: Any, context: ToolContext) -> ToolResult:
        return ToolResult(
            ok=False,
            output="",
            error_code=ToolErrorCode.UNAVAILABLE,
            latency_ms=2,
        )


class SlowTool(FakePythonTool):
    """Answers with a sandbox-side timeout (E_TIMEOUT, kept out of the C pool)."""

    async def execute(self, arguments: Any, context: ToolContext) -> ToolResult:
        return ToolResult(
            ok=False,
            output="stdout:\npartial\n",
            error_code=ToolErrorCode.TIMEOUT,
            latency_ms=900,
            metadata={"execution_time": 10.0, "sandbox_status": "TimeLimitExceeded"},
        )


def tool_call(code: str) -> str:
    return f'<tool_call>{{"name":"python","arguments":{{"code":"{code}"}}}}</tool_call>'


def final(answer: str) -> str:
    return f'<final>{{"answer":"{answer}"}}</final>'
