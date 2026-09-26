"""Bounded symbolic tool executed in a dedicated child process."""

import asyncio
import multiprocessing as mp
import queue
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.tools.base import ToolContext, ToolErrorCode, ToolResult

MAX_EXPR_CHARS = 8192
MAX_NESTING = 64
MAX_POWER = 10000
# Child-process deadline.  Spawn must re-import sympy from scratch; on a
# loaded shared host that can take several seconds, and rollout steps spawn
# several workers concurrently (one per batch row), so cold imports contend
# and 10s misreported correct calculations as timeouts (observed on the
# 2026-09-24 contract run: 4/4 concurrent cold spawns tripped the deadline).
# 30s keeps generous headroom; the join returns as soon as the child exits,
# so the deadline only bounds genuinely hung calculations.
WORKER_TIMEOUT_SECONDS = 30.0
_OPERATIONS = Literal["simplify", "factor", "expand", "solve", "diff", "integrate", "numeric"]
_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
_POWER = re.compile(r"\^\s*(\d+)")


class SympyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: _OPERATIONS
    expression: str = Field(min_length=1, max_length=MAX_EXPR_CHARS)
    variables: list[str] = Field(default_factory=list, max_length=4)
    lower: str | None = None
    upper: str | None = None


class SympyTool:
    name = "sympy"
    description = "Safely simplify, factor, expand, solve, differentiate, integrate, or evaluate an expression."
    arguments_model = SympyArguments

    async def execute(self, arguments: SympyArguments, context: ToolContext) -> ToolResult:
        return await asyncio.to_thread(_run_isolated, arguments)


def _run_isolated(arguments: SympyArguments) -> ToolResult:
    rejected = _reject(arguments)
    if rejected is not None:
        return ToolResult(ok=False, output="", error_code=ToolErrorCode.INVALID_ARGUMENTS, latency_ms=0, metadata={"reason": rejected})
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(arguments.model_dump(), result_queue), daemon=True)
    process.start()
    process.join(WORKER_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join()
        return ToolResult(ok=False, output="", error_code=ToolErrorCode.TIMEOUT, latency_ms=int(WORKER_TIMEOUT_SECONDS * 1000))
    try:
        # The child may have exited just before its queue feeder flushes.  A
        # tiny bounded wait avoids turning a successful calculation into a
        # spurious tool failure, without extending the execution deadline.
        payload = result_queue.get(timeout=0.1)
    except queue.Empty:
        return ToolResult(ok=False, output="", error_code=ToolErrorCode.EXECUTION_ERROR, latency_ms=0)
    return ToolResult.model_validate(payload)


def _worker(values: dict[str, object], result_queue: Any) -> None:
    import sympy

    try:
        args = SympyArguments.model_validate(values)
        symbols = {name: sympy.Symbol(name) for name in args.variables}
        locals_map = {**symbols, "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "sqrt": sympy.sqrt, "pi": sympy.pi, "E": sympy.E}
        expression = sympy.sympify(args.expression.replace("^", "**"), locals=locals_map)
        if args.operation == "factor":
            output = sympy.factor(expression)
        elif args.operation == "expand":
            output = sympy.expand(expression)
        elif args.operation == "simplify":
            output = sympy.simplify(expression)
        elif args.operation == "solve":
            variable = symbols[args.variables[0]]
            output = sorted(sympy.solve(expression, variable), key=str)
        elif args.operation == "diff":
            output = sympy.diff(expression, symbols[args.variables[0]])
        elif args.operation == "integrate":
            variable = symbols[args.variables[0]]
            output = sympy.integrate(expression, (variable, args.lower, args.upper)) if args.lower is not None and args.upper is not None else sympy.integrate(expression, variable)
        else:
            output = sympy.N(expression, 50)
        result_queue.put(ToolResult(ok=True, output=str(output), latency_ms=0, metadata={"latex": sympy.latex(output)}).model_dump())
    # This is an isolation boundary for third-party symbolic code.  The parent
    # receives only a structured result, never an arbitrary child exception.
    except Exception:  # noqa: BLE001
        result_queue.put(ToolResult(ok=False, output="", error_code=ToolErrorCode.EXECUTION_ERROR, latency_ms=0).model_dump())


def _reject(arguments: SympyArguments) -> str | None:
    text = arguments.expression
    if any(token in text for token in ("__", ".", "[", "]", "{", "}", "'", '"', "import")):
        return "unsafe syntax"
    if text.count("(") != text.count(")") or max(_nesting(text), default=0) > MAX_NESTING:
        return "invalid nesting"
    if any(int(power) > MAX_POWER for power in _POWER.findall(text)):
        return "power exceeds limit"
    allowed_functions = {"sin", "cos", "tan", "sqrt"}
    constants = {"pi", "E"}
    for identifier in _IDENTIFIER.finditer(text):
        name = identifier.group(0)
        if name in allowed_functions | constants | set(arguments.variables):
            continue
        # Bare identifiers denote local symbols (for example ``x`` in a
        # factorization).  Identifiers used as calls must be from our allowlist.
        if text[identifier.end() :].lstrip().startswith("("):
            return "unknown function"
    if arguments.operation in {"solve", "diff", "integrate"} and not arguments.variables:
        return "operation requires a variable"
    return None


def _nesting(text: str) -> list[int]:
    depth = 0
    values: list[int] = []
    for char in text:
        if char == "(":
            depth += 1
            values.append(depth)
        elif char == ")":
            depth -= 1
    return values
