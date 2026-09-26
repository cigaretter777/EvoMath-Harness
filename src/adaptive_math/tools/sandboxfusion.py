"""Small, fail-closed client for the external Python execution service."""

import os
import time
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from adaptive_math.core.types import JSONValue
from adaptive_math.tools.base import ToolErrorCode, ToolResult

_DEADLINE_MARGIN_SECONDS = 1.0


class SandboxRunRequest(BaseModel):
    """The intentionally narrow request accepted by SandboxFusion."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=20_000)
    language: Literal["python"] = "python"
    run_timeout: float | None = Field(default=None, gt=0)


class SandboxRunResult(BaseModel):
    """Execution payload nested under SandboxFusion's ``run_result`` key."""

    model_config = ConfigDict(extra="forbid")

    status: str
    # Every field except status is nullable in SandboxFusion's own OpenAPI
    # (CommandRunResult): a killed or failed run answers without them.
    execution_time: float | None = Field(default=None, ge=0)
    return_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None


class SandboxRunResponse(BaseModel):
    """The response envelope returned by SandboxFusion's ``/run_code`` API."""

    model_config = ConfigDict(extra="ignore")

    status: str
    run_result: SandboxRunResult | None = None


class SandboxFusionClient:
    """Execute code remotely; errors never trigger a local fallback."""

    def __init__(self, base_url: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        resolved_url = base_url or os.environ.get("ADAPTIVE_MATH_SANDBOX_URL", "http://127.0.0.1:8080")
        self._client = client or httpx.AsyncClient(
            base_url=resolved_url.rstrip("/"), timeout=httpx.Timeout(6.0, connect=1.0)
        )
        self._owns_client = client is None

    async def run_code(self, code: str, timeout_seconds: float | None = None) -> ToolResult:
        request_payload = SandboxRunRequest(code=code, run_timeout=timeout_seconds).model_dump(
            exclude_none=True
        )
        started = time.perf_counter()
        try:
            response = await self._post(request_payload, timeout_seconds)
        except httpx.TimeoutException:
            return ToolResult(
                ok=False,
                output="",
                error_code=ToolErrorCode.TIMEOUT,
                latency_ms=_elapsed_ms(started),
            )
        except httpx.HTTPError:
            return ToolResult(
                ok=False,
                output="",
                error_code=ToolErrorCode.UNAVAILABLE,
                latency_ms=_elapsed_ms(started),
            )

        if response.status_code >= 500:
            return ToolResult(
                ok=False,
                output="",
                error_code=ToolErrorCode.UNAVAILABLE,
                latency_ms=_elapsed_ms(started),
            )
        try:
            payload = SandboxRunResponse.model_validate(response.json())
        except (ValidationError, ValueError):
            return ToolResult(
                ok=False,
                output="",
                error_code=ToolErrorCode.EXECUTION_ERROR,
                latency_ms=_elapsed_ms(started),
            )

        if payload.run_result is None:
            return ToolResult(
                ok=False,
                output="",
                error_code=ToolErrorCode.EXECUTION_ERROR,
                latency_ms=_elapsed_ms(started),
            )
        run_result = payload.run_result
        output = _format_output(run_result.stdout, run_result.stderr)
        if run_result.status == "TimeLimitExceeded":
            return ToolResult(
                ok=False,
                output=output,
                error_code=ToolErrorCode.TIMEOUT,
                latency_ms=_elapsed_ms(started),
                metadata=_timing_metadata(run_result.execution_time),
            )
        if payload.status == "Success" and run_result.status == "Finished" and run_result.return_code == 0:
            return ToolResult(
                ok=True,
                output=output,
                latency_ms=_elapsed_ms(started),
                metadata=_timing_metadata(run_result.execution_time),
            )
        return ToolResult(
            ok=False,
            output=output,
            error_code=ToolErrorCode.EXECUTION_ERROR,
            latency_ms=_elapsed_ms(started),
            metadata=_timing_metadata(run_result.execution_time),
        )

    async def _post(self, payload: dict[str, object], timeout_seconds: float | None) -> httpx.Response:
        """An unbounded request must omit the argument: httpx reads ``timeout=None``
        as "disable every timeout", not "keep using the client default". A bounded
        one gets a local deadline just past the remote one, so a real
        TimeLimitExceeded answer is observed instead of a blind socket timeout."""
        if timeout_seconds is None:
            return await self._client.post("/run_code", json=payload)
        return await self._client.post(
            "/run_code",
            json=payload,
            timeout=httpx.Timeout(timeout_seconds + _DEADLINE_MARGIN_SECONDS, connect=1.0),
        )

    async def health(self) -> bool:
        try:
            response = await self._client.get("/v1/ping")
        except httpx.HTTPError:
            return False
        return response.is_success

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _format_output(stdout: str | None, stderr: str | None) -> str:
    sections: list[str] = []
    if stdout:
        sections.append(f"stdout:\n{stdout}")
    if stderr:
        sections.append(f"stderr:\n{stderr}")
    return "\n".join(sections)


def _timing_metadata(execution_time: float | None) -> dict[str, JSONValue]:
    """Omit the key instead of writing null: the budget reads it as seconds."""
    return {} if execution_time is None else {"execution_time": execution_time}


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))
