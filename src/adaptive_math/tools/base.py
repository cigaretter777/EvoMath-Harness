"""Non-executable interfaces shared by all agent tools."""

import asyncio
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.core.types import JSONValue


class ToolErrorCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    OUTPUT_LIMIT = "output_limit"
    UNAVAILABLE = "unavailable"


class ToolExecutionError(Exception):
    """A declared, recoverable tool-operation failure."""


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    output: str
    error_code: ToolErrorCode | None = None
    latency_ms: int = Field(ge=0)
    truncated: bool = False
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ToolContext(BaseModel):
    """Public per-call context; never carries a label or verifier handle."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trace_id: str
    task_id: str
    remaining_observation_chars: int = Field(gt=0)
    remaining_python_seconds: float | None = None
    cancellation: asyncio.Event | None = None


class Tool(Protocol):
    name: str
    description: str
    arguments_model: type[BaseModel]

    async def execute(self, arguments: BaseModel, context: ToolContext) -> ToolResult: ...
