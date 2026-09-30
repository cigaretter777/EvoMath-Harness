import asyncio

from adaptive_math.tools.base import ToolContext
from adaptive_math.tools.sympy_tool import SympyArguments, SympyTool


def context() -> ToolContext:
    return ToolContext(trace_id="trace", task_id="task", remaining_observation_chars=1000)


def test_sympy_tool_factors_and_sorts_solutions_deterministically() -> None:
    tool = SympyTool()

    factor = asyncio.run(tool.execute(SympyArguments(operation="factor", expression="x^2 - 1"), context()))
    solve = asyncio.run(tool.execute(SympyArguments(operation="solve", expression="x^2 - 4", variables=["x"]), context()))

    assert factor.ok and factor.output == "(x - 1)*(x + 1)"
    assert solve.ok and solve.output == "[-2, 2]"


def test_sympy_tool_supports_calculus() -> None:
    tool = SympyTool()

    derivative = asyncio.run(tool.execute(SympyArguments(operation="diff", expression="sin(x)", variables=["x"]), context()))
    integral = asyncio.run(tool.execute(SympyArguments(operation="integrate", expression="x", variables=["x"], lower="0", upper="2"), context()))

    assert derivative.output == "cos(x)"
    assert integral.output == "2"


def test_sympy_tool_rewrites_equation_markers_for_solve() -> None:
    tool = SympyTool()

    # Model-emitted equations arrive with "==" (or a bare "="); sympify alone
    # parses that as Python equality and returns a plain False, so solve used
    # to yield [] with ok=True for every equation-shaped call (2026-09-29).
    solve_double = asyncio.run(
        tool.execute(SympyArguments(operation="solve", expression="1/6 + 1/3 - 1/x == 0", variables=["x"]), context())
    )
    solve_single = asyncio.run(
        tool.execute(SympyArguments(operation="solve", expression="x^2 = 4", variables=["x"]), context())
    )

    assert solve_double.ok and solve_double.output == "[2]"
    assert solve_single.ok and solve_single.output == "[-2, 2]"
