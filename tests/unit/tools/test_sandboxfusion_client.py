import asyncio
import json

import httpx
import pytest

from adaptive_math.tools.base import ToolErrorCode
from adaptive_math.tools.sandboxfusion import SandboxFusionClient


def test_run_code_posts_only_code_and_language(respx_mock: object) -> None:
    route = respx_mock.post("http://sandbox.test/run_code").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "Success",
                "message": "",
                "compile_result": None,
                "run_result": {
                    "status": "Finished",
                    "execution_time": 0.12,
                    "return_code": 0,
                    "stdout": "42\n",
                    "stderr": "",
                },
            },
        )
    )
    client = SandboxFusionClient(base_url="http://sandbox.test")

    result = asyncio.run(client.run_code("print(42)"))

    assert route.called
    assert json.loads(route.calls[0].request.content) == {"code": "print(42)", "language": "python"}
    assert result.ok and result.output == "stdout:\n42\n"
    asyncio.run(client.aclose())


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            httpx.Response(
                200,
                json={
                    "status": "Failed",
                    "message": "",
                    "compile_result": None,
                    "run_result": {
                        "status": "Finished",
                        "execution_time": 0.1,
                        "return_code": 1,
                        "stdout": "",
                        "stderr": "SyntaxError",
                    },
                },
            ),
            ToolErrorCode.EXECUTION_ERROR,
        ),
        (httpx.Response(500), ToolErrorCode.UNAVAILABLE),
        (httpx.Response(200, json={"unexpected": "shape"}), ToolErrorCode.EXECUTION_ERROR),
        # Captured from the live SandboxFusion when run_timeout fires: the
        # sandbox itself rejects an unset return code, so the client must too.
        (
            httpx.Response(
                200,
                json={
                    "status": "Failed",
                    "message": "",
                    "compile_result": None,
                    "run_result": {
                        "status": "TimeLimitExceeded",
                        "execution_time": 10.0015,
                        "return_code": None,
                        "stdout": "",
                        "stderr": "",
                    },
                },
            ),
            ToolErrorCode.TIMEOUT,
        ),
    ],
)
def test_run_code_maps_remote_failures(
    respx_mock: object, response: httpx.Response, expected: ToolErrorCode
) -> None:
    respx_mock.post("http://sandbox.test/run_code").mock(return_value=response)
    client = SandboxFusionClient(base_url="http://sandbox.test")

    result = asyncio.run(client.run_code("raise ValueError()"))

    assert not result.ok
    assert result.error_code is expected
    asyncio.run(client.aclose())


def test_run_code_bounds_the_remote_run_when_a_timeout_is_given(respx_mock: object) -> None:
    """The sandbox must stop on the agent's remaining budget, not on its own
    default, and the local deadline must outlast the remote one so the
    TimeLimitExceeded answer — with its execution_time — is what comes back."""
    route = respx_mock.post("http://sandbox.test/run_code").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "Failed",
                "message": "",
                "compile_result": None,
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "execution_time": 3.0,
                    "return_code": None,
                    "stdout": "",
                    "stderr": "",
                },
            },
        )
    )
    client = SandboxFusionClient(base_url="http://sandbox.test")

    result = asyncio.run(client.run_code("while True: pass", timeout_seconds=3.0))

    assert json.loads(route.calls[0].request.content) == {
        "code": "while True: pass",
        "language": "python",
        "run_timeout": 3.0,
    }
    assert route.calls[0].request.extensions["timeout"]["read"] > 3.0
    assert result.error_code is ToolErrorCode.TIMEOUT
    asyncio.run(client.aclose())


def test_run_code_omits_an_unset_timeout_so_the_server_default_still_applies(respx_mock: object) -> None:
    route = respx_mock.post("http://sandbox.test/run_code").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "Success",
                "message": "",
                "compile_result": None,
                "run_result": {
                    "status": "Finished",
                    "execution_time": 0.1,
                    "return_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                },
            },
        )
    )
    client = SandboxFusionClient(base_url="http://sandbox.test")

    asyncio.run(client.run_code("print('ok')"))

    assert json.loads(route.calls[0].request.content) == {"code": "print('ok')", "language": "python"}
    # httpx reads timeout=None as "no timeout at all", so the client default must
    # survive by omitting the argument, not by passing None.
    assert route.calls[0].request.extensions["timeout"] == {
        "connect": 1.0,
        "read": 6.0,
        "write": 6.0,
        "pool": 6.0,
    }
    asyncio.run(client.aclose())


def test_health_probes_the_route_sandboxfusion_serves(respx_mock: object) -> None:
    """SandboxFusion answers on /v1/ping; probing /health reports a live sandbox
    as unavailable."""
    ping = respx_mock.get("http://sandbox.test/v1/ping").mock(
        return_value=httpx.Response(200, json="pong")
    )
    client = SandboxFusionClient(base_url="http://sandbox.test")

    assert asyncio.run(client.health()) is True
    assert ping.called
    asyncio.run(client.aclose())


def test_run_code_accepts_the_optional_fields_of_a_failed_command_result(respx_mock: object) -> None:
    """CommandRunResult requires only ``status``; a compile-stage failure answers
    with the rest absent or null and must not be mistaken for a bad contract."""
    respx_mock.post("http://sandbox.test/run_code").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "Failed",
                "message": "compile failed",
                "compile_result": {"status": "Error"},
                "run_result": {
                    "status": "Error",
                    "execution_time": None,
                    "return_code": None,
                    "stdout": None,
                    "stderr": None,
                },
            },
        )
    )
    client = SandboxFusionClient(base_url="http://sandbox.test")

    result = asyncio.run(client.run_code("print(1)"))

    assert result.error_code is ToolErrorCode.EXECUTION_ERROR
    assert result.metadata == {}
    asyncio.run(client.aclose())


def test_timed_out_run_still_reports_execution_time_for_the_python_budget(respx_mock: object) -> None:
    """A killed run consumed real sandbox time; charging it stops the agent from
    burning an unbudgeted number of slow attempts."""
    respx_mock.post("http://sandbox.test/run_code").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "Failed",
                "message": "",
                "compile_result": None,
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "execution_time": 10.0015,
                    "return_code": None,
                    "stdout": "",
                    "stderr": "",
                },
            },
        )
    )
    client = SandboxFusionClient(base_url="http://sandbox.test")

    result = asyncio.run(client.run_code("while True: pass"))

    assert result.error_code is ToolErrorCode.TIMEOUT
    assert result.metadata["execution_time"] == 10.0015
    asyncio.run(client.aclose())
