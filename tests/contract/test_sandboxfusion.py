"""Contract against a real SandboxFusion instance.

Skipped unless ``ADAPTIVE_MATH_RUN_SANDBOX_CONTRACT=1`` and
``ADAPTIVE_MATH_SANDBOX_URL`` points at a live sandbox. These are the assertions
a mocked client cannot make: that the served routes match what the client asks
for, and that the remote deadline — not the local socket — ends a slow run.

Each case drives one event loop: a real ``httpx.AsyncClient`` binds its pool to
the loop that created it, so closing it from a second ``asyncio.run`` fails.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable

import pytest

from adaptive_math.tools.base import ToolErrorCode, ToolResult
from adaptive_math.tools.sandboxfusion import SandboxFusionClient

pytestmark = pytest.mark.skipif(
    os.environ.get("ADAPTIVE_MATH_RUN_SANDBOX_CONTRACT") != "1",
    reason="set ADAPTIVE_MATH_RUN_SANDBOX_CONTRACT=1 against a live SandboxFusion",
)


def _run[T](body: Callable[[SandboxFusionClient], Awaitable[T]]) -> T:
    async def main() -> T:
        client = SandboxFusionClient()
        try:
            return await body(client)
        finally:
            await client.aclose()

    return asyncio.run(main())


def test_live_sandbox_executes_python_and_reports_its_own_timing() -> None:
    result: ToolResult = _run(lambda client: client.run_code("print(6 * 7)"))

    assert result.ok, result.error_code
    assert "42" in result.output
    # The budget is normalized against this number, so it must come from the
    # sandbox rather than being inferred locally.
    assert result.metadata["execution_time"] > 0


def test_live_health_probe_matches_the_route_the_sandbox_serves() -> None:
    """SandboxFusion answers liveness on /v1/ping; a probe aimed at any other
    path reports a healthy sandbox as dead."""
    assert _run(lambda client: client.health()) is True


def test_live_run_is_killed_by_the_remote_deadline_and_charged_for_it() -> None:
    result: ToolResult = _run(
        lambda client: client.run_code("import time\ntime.sleep(30)", timeout_seconds=1.5)
    )

    assert result.error_code is ToolErrorCode.TIMEOUT
    # A local socket timeout carries no execution_time at all; only the sandbox's
    # own TimeLimitExceeded answer reports how long it actually ran.
    assert result.metadata["execution_time"] == pytest.approx(1.5, abs=0.6)


def test_live_unreachable_sandbox_stays_fail_closed() -> None:
    """No fallback to host execution, ever: the model's code must not run here."""
    async def main() -> ToolResult:
        client = SandboxFusionClient(base_url="http://127.0.0.1:9")
        try:
            return await client.run_code("print(1)")
        finally:
            await client.aclose()

    result = asyncio.run(main())

    assert not result.ok
    assert result.error_code is ToolErrorCode.UNAVAILABLE
    assert result.output == ""
