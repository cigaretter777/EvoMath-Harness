import asyncio

from adaptive_math.tools.base import ToolContext, ToolErrorCode, ToolResult
from adaptive_math.tools.python_tool import PythonArguments, PythonTool


class RecordingSandbox:
    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.code: str | None = None
        self.timeout_seconds: float | None = None

    async def run_code(self, code: str, timeout_seconds: float | None = None) -> ToolResult:
        self.code = code
        self.timeout_seconds = timeout_seconds
        return self.result


def _context(**overrides: object) -> ToolContext:
    payload: dict[str, object] = {
        "trace_id": "trace",
        "task_id": "task",
        "remaining_observation_chars": 10,
    }
    payload.update(overrides)
    return ToolContext(**payload)  # type: ignore[arg-type]


def test_python_tool_bounds_the_remote_run_by_the_remaining_python_budget() -> None:
    """The remote sandbox keeps running after the local client gives up, so the
    agent's remaining seconds must travel with the request."""
    sandbox = RecordingSandbox(ToolResult(ok=True, output="ok", latency_ms=7))
    tool = PythonTool(sandbox)

    asyncio.run(
        tool.execute(
            PythonArguments(code="print(1)"), _context(remaining_python_seconds=4.5)
        )
    )

    assert sandbox.timeout_seconds == 4.5


def test_python_tool_leaves_the_remote_default_when_no_budget_is_reported() -> None:
    sandbox = RecordingSandbox(ToolResult(ok=True, output="ok", latency_ms=7))
    tool = PythonTool(sandbox)

    asyncio.run(tool.execute(PythonArguments(code="print(1)"), _context()))

    assert sandbox.timeout_seconds is None


def test_python_tool_delegates_only_code_to_sandbox_and_bounds_observation() -> None:
    sandbox = RecordingSandbox(ToolResult(ok=True, output="12345678901", latency_ms=7))
    tool = PythonTool(sandbox)

    result = asyncio.run(tool.execute(PythonArguments(code="print(6 * 7)"), _context()))

    assert sandbox.code == "print(6 * 7)"
    assert result.ok
    assert result.output == "1234567890"
    assert result.truncated


def test_python_tool_propagates_structured_sandbox_failure() -> None:
    tool = PythonTool(
        RecordingSandbox(
            ToolResult(ok=False, output="", error_code=ToolErrorCode.UNAVAILABLE, latency_ms=3)
        )
    )

    result = asyncio.run(tool.execute(PythonArguments(code="print(1)"), _context()))

    assert not result.ok
    assert result.error_code is ToolErrorCode.UNAVAILABLE
