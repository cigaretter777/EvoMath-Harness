"""Python tool facade that can execute code only through SandboxFusion."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.tools.base import ToolContext, ToolResult


class PythonArguments(BaseModel):
    """No paths, environment, networking or container controls are exposed."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=20_000)


class SandboxExecutor(Protocol):
    async def run_code(
        self, code: str, timeout_seconds: float | None = None
    ) -> ToolResult: ...


class PythonTool:
    """Delegate untrusted model code to the configured remote sandbox."""

    name = "python"
    description = "Execute a short Python program in the isolated SandboxFusion service."
    arguments_model = PythonArguments

    def __init__(self, sandbox: SandboxExecutor) -> None:
        self._sandbox = sandbox

    async def execute(self, arguments: PythonArguments, context: ToolContext) -> ToolResult:
        result = await self._sandbox.run_code(
            arguments.code, timeout_seconds=_remote_timeout(context)
        )
        return _truncate(result, context.remaining_observation_chars)


def _remote_timeout(context: ToolContext) -> float | None:
    """The environment still owns budget enforcement; this only bounds how long
    the sandbox may keep computing after the agent has moved on. A non-positive
    remainder is treated as "no bound reported"."""
    remaining = context.remaining_python_seconds
    return remaining if remaining is not None and remaining > 0 else None


def _truncate(result: ToolResult, limit: int) -> ToolResult:
    encoded = result.output.encode()
    if len(encoded) <= limit:
        return result
    return result.model_copy(
        update={"output": encoded[:limit].decode(errors="ignore"), "truncated": True}
    )
