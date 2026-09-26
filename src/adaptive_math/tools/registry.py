"""Validation and bounded observation handling for registered tools."""

from collections.abc import Iterable

from pydantic import ValidationError

from adaptive_math.tools.base import (
    Tool,
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolResult,
)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    async def execute(
        self, name: str, arguments: object, context: ToolContext
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return _error(ToolErrorCode.UNKNOWN_TOOL)
        try:
            validated = tool.arguments_model.model_validate(arguments)
        except ValidationError:
            return _error(ToolErrorCode.INVALID_ARGUMENTS)
        try:
            result = await tool.execute(validated, context)
        except TimeoutError:
            return _error(ToolErrorCode.TIMEOUT)
        except OSError:
            return _error(ToolErrorCode.UNAVAILABLE)
        except ToolExecutionError:
            return _error(ToolErrorCode.EXECUTION_ERROR)
        return _truncate(result, context.remaining_observation_chars)

    @property
    def names(self) -> tuple[str, ...]:
        """Registered tool names, sorted — the environment's enabled tool set."""
        return tuple(sorted(self._tools))

    def descriptions(self) -> list[dict[str, object]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "arguments_schema": tool.arguments_model.model_json_schema(),
            }
            for _, tool in sorted(self._tools.items())
        ]


def _error(code: ToolErrorCode) -> ToolResult:
    return ToolResult(ok=False, output="", error_code=code, latency_ms=0)


def _truncate(result: ToolResult, limit: int) -> ToolResult:
    encoded = result.output.encode()
    if len(encoded) <= limit:
        return result
    output = encoded[:limit].decode(errors="ignore")
    return result.model_copy(update={"output": output, "truncated": True})
