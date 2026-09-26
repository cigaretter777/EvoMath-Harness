import asyncio

import pytest
from pydantic import BaseModel, Field

from adaptive_math.tools.base import ToolContext, ToolErrorCode, ToolExecutionError, ToolResult
from adaptive_math.tools.registry import ToolRegistry


class EchoArguments(BaseModel):
    text: str = Field(min_length=1)


class EchoTool:
    name = "echo"
    description = "Echo input text."
    arguments_model = EchoArguments

    async def execute(self, arguments: EchoArguments, context: ToolContext) -> ToolResult:
        return ToolResult(ok=True, output=arguments.text, latency_ms=0)


def make_context(chars: int = 100) -> ToolContext:
    return ToolContext(trace_id="trace", task_id="task", remaining_observation_chars=chars)


def test_registry_validates_arguments_and_executes_registered_tool() -> None:
    result = asyncio.run(ToolRegistry([EchoTool()]).execute("echo", {"text": "hello"}, make_context()))

    assert result == ToolResult(ok=True, output="hello", latency_ms=0)


def test_registry_returns_structured_errors_for_unknown_or_invalid_calls() -> None:
    registry = ToolRegistry([EchoTool()])

    unknown = asyncio.run(registry.execute("missing", {}, make_context()))
    invalid = asyncio.run(registry.execute("echo", {"text": ""}, make_context()))
    assert unknown.error_code is ToolErrorCode.UNKNOWN_TOOL
    assert invalid.error_code is ToolErrorCode.INVALID_ARGUMENTS


def test_registry_rejects_duplicate_names_and_truncates_utf8_safely() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([EchoTool(), EchoTool()])

    result = asyncio.run(ToolRegistry([EchoTool()]).execute("echo", {"text": "你好世界"}, make_context(5)))
    assert result.output == "你"
    assert result.truncated


def test_registry_exposes_the_same_json_schema_used_for_validation() -> None:
    descriptions = ToolRegistry([EchoTool()]).descriptions()

    assert descriptions == [
        {"name": "echo", "description": "Echo input text.", "arguments_schema": EchoArguments.model_json_schema()}
    ]


def test_registry_exposes_sorted_tool_names() -> None:
    assert ToolRegistry([EchoTool()]).names == ("echo",)
    assert ToolRegistry([]).names == ()


def test_registry_maps_declared_operational_failures_without_swallowing_programming_errors() -> None:
    class FailingTool(EchoTool):
        async def execute(self, arguments: EchoArguments, context: ToolContext) -> ToolResult:
            raise ToolExecutionError

    result = asyncio.run(ToolRegistry([FailingTool()]).execute("echo", {"text": "x"}, make_context()))
    assert result.error_code is ToolErrorCode.EXECUTION_ERROR
