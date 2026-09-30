"""The compiler proposes; the tool and the verifier dispose.

Every assertion here is about one of the two rules the module promises: the
problem text is the only input, and a proposed call must fall inside the
grammar ``SympyTool`` is willing to execute.
"""

import pytest
import sympy

from adaptive_math.tools.sympy_tool import SympyArguments, _reject
from adaptive_math.training.sympy_compiler import (
    Compilation,
    compile_sympy_call,
    compile_sympy_candidates,
)


def compile_or_fail(problem: str) -> SympyArguments:
    compilation = compile_sympy_call(problem)
    assert compilation.arguments is not None, compilation.reason
    return compilation.arguments


def parses_like_the_worker(arguments: SympyArguments) -> bool:
    """Mirror the parse step of ``sympy_tool._worker`` for one proposed call.

    Driving the real tool means spawning a child process, which a unit test
    should not pay for.  The compiler's obligation is narrower than the tool's:
    it must not propose a call whose first line the worker cannot read.
    """
    locals_map = {
        **{name: sympy.Symbol(name) for name in arguments.variables},
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "sqrt": sympy.sqrt,
        "pi": sympy.pi,
        "E": sympy.E,
    }
    try:
        sympy.sympify(arguments.expression.replace("^", "**"), locals=locals_map)
    except Exception:  # noqa: BLE001 - the worker swallows every exception here
        return False
    return True


# Problems the compiler must handle, reused by the invariant test below.
BOUNDARY_PROBLEMS = [
    r"Find the roots of $x^2 - 5x + 6 = 0$.",
    r"Expand $(x+1)^3$.",
    r"Factor $x^2 - 4$.",
    r"Solve for $x$: $\frac{x}{2} + 3 = 7$.",
    r"What is $24 \times 37$?",
    r"Evaluate $\sqrt{17}$.",
    r"Find the derivative of $x^3 + 2x$.",
    r"Compute $1 + 1$.",
    r"Find the integral of $3x^2$.",
    r"Simplify $\frac{\frac{1}{2} + \frac{1}{3}}{\frac{1}{6}}$.",
    r"Expand $(x+1)(x-1)$.",
]

# Every span here passed ``SympyTool._reject`` and then died inside the worker,
# so a scan scored them as tool errors rather than as wrong answers.  Captured
# from a 200-problem run; each is one construct the tool's safety boundary
# admits and its parser does not.
PARSER_INVALID_PROBLEMS = [
    r"Evaluate $3;3$.",
    r"Solve $0<x<1$.",
    r"Evaluate $=1:2$.",
    r"Expand $x^7-()$.",
    r"Solve $|x^2-3x+2|+|x^2+2x-3|=11$.",
    r"Solve $x+|x|=0$.",
    r"Expand $f(z)=\frac{z}{z^2-2z-3}$.",
]


def test_compiled_call_is_always_within_the_tools_safety_boundary() -> None:
    """``_reject`` is the tool's own gate; a proposal it refuses is wasted."""
    for problem in BOUNDARY_PROBLEMS:
        arguments = compile_or_fail(problem)
        assert _reject(arguments) is None, problem


@pytest.mark.parametrize(
    ("problem", "reason"),
    [
        # A character the parser has no meaning for.
        (r"Evaluate $3;3$.", "unconvertible_span"),
        (r"Solve $0<x<1$.", "unconvertible_span"),
        (r"Evaluate $=1:2$.", "unconvertible_span"),
        # ``|x|`` is an absolute value to a reader and a syntax error to
        # ``sympify``; the tool's grammar has no way to say it.
        (r"Solve $|x^2-3x+2|+|x^2+2x-3|=11$.", "unconvertible_span"),
        (r"Solve $x+|x|=0$.", "unconvertible_span"),
        # An operator with no operand: ``x^7-()``.
        (r"Expand $x^7-()$.", "unconvertible_span"),
        # An equation under an operation that cannot consume one.  The span
        # scores highest of all (it holds an ``=``) and still yields no call,
        # which is why the ``=`` rule has to live in ``_build`` and not in the
        # ranking.
        (r"Expand $f(z)=\frac{z}{z^2-2z-3}$.", "unbuildable_expand"),
    ],
)
def test_a_parser_invalid_span_is_refused_instead_of_proposed(
    problem: str, reason: str
) -> None:
    compilation = compile_sympy_call(problem)

    assert compilation.arguments is None
    assert compilation.reason == reason


@pytest.mark.parametrize(
    ("problem", "reason"),
    [
        (r"Factor $x^2 - 4 = 0$.", "unbuildable_factor"),
        (r"Simplify $x^2 - 4 = 0$.", "unbuildable_simplify"),
        (r"Evaluate $x^2 - 4 = 0$.", "unbuildable_numeric"),
    ],
)
def test_an_equation_is_never_handed_to_an_operation_that_cannot_use_one(
    problem: str, reason: str
) -> None:
    """Only ``solve`` consumes an equation; the rest would compute nothing."""
    compilation = compile_sympy_call(problem)

    assert compilation.arguments is None
    assert compilation.reason == reason


def test_every_proposed_call_survives_the_workers_parse_step() -> None:
    """The invariant the regressions above are instances of.

    Safety is not parseability: the corpus keeps producing spans that clear
    ``_reject`` and then raise inside the worker, and each one costs a tool
    error the gate has to explain away.
    """
    for problem in BOUNDARY_PROBLEMS + PARSER_INVALID_PROBLEMS:
        for candidate in compile_sympy_candidates(problem, limit=4):
            if candidate.arguments is None:
                continue
            assert parses_like_the_worker(candidate.arguments), (
                problem,
                candidate.arguments.expression,
            )


@pytest.mark.parametrize(
    ("problem", "expression"),
    [
        # Juxtaposition is multiplication in LaTeX but a SyntaxError to
        # ``sympify``: the tool cannot parse ``3x``.
        (r"Find the integral of $3x^2$.", "3*x^2"),
        (r"Find the derivative of $x^3 + 2x$.", "x^3+2*x"),
        (r"Expand $(x+1)(x-1)$.", "(x+1)*(x-1)"),
        (r"Expand $2(x+1)$.", "2*(x+1)"),
        # Macros with a fixed meaning.
        (r"What is $24 \times 37$?", "24*37"),
        (r"What is $24 \cdot 37$?", "24*37"),
        (r"What is $24 \div 4$?", "24/4"),
        (r"What is $2\pi$?", "2*pi"),
        (r"Evaluate $\sqrt{17}$.", "sqrt(17)"),
        (r"Simplify $\frac{1}{2} + \frac{1}{3}$.", "((1)/(2))+((1)/(3))"),
    ],
)
def test_latex_is_translated_into_the_grammar_the_tool_parses(
    problem: str, expression: str
) -> None:
    assert compile_or_fail(problem).expression == expression


def test_nested_fraction_pairs_its_braces_instead_of_crossing_them() -> None:
    """Regression: the first ``\\frac`` match was consumed as the numerator.

    Reading the outermost macro first pairs the denominator of the inner
    fraction with the numerator of the outer one, which compiled a different
    expression than the problem asked for.
    """
    arguments = compile_or_fail(r"Simplify $\frac{\frac{1}{2} + \frac{1}{3}}{\frac{1}{6}}$.")

    assert arguments.expression == "((((1)/(2))+((1)/(3)))/(((1)/(6))))"
    assert arguments.operation == "simplify"


def test_fraction_does_not_consume_its_own_denominator_as_a_numerator() -> None:
    """Regression: re-reading the numerator at ``end`` swallowed the denominator."""
    arguments = compile_or_fail(r"Solve for $x$: $\frac{x}{2} + 3 = 7$.")

    assert arguments.expression == "(((x)/(2))+3)-(7)"
    assert arguments.variables == ["x"]


def test_two_character_symbols_survive_juxtaposition() -> None:
    """Regression: ``\\b`` hides the ``x`` in ``3x``, so no variable was found."""
    arguments = compile_or_fail(r"Find the derivative of $3x^2$.")

    assert arguments.variables == ["x"]


def test_solve_reduces_an_equation_so_the_roots_stay_the_roots() -> None:
    arguments = compile_or_fail(r"Find the roots of $x^2 - 5x + 6 = 0$.")

    assert arguments.operation == "solve"
    assert arguments.expression == "(x^2-5*x+6)-(0)"

    no_equation = compile_or_fail(r"Find the roots of $x^2 - 5x + 6$.")
    assert no_equation.expression == "x^2-5*x+6"


@pytest.mark.parametrize(
    ("problem", "operation"),
    [
        (r"Find the derivative of $x^3$.", "diff"),
        (r"Differentiate $x^3$.", "diff"),
        (r"Find the integral of $3x^2$.", "integrate"),
        (r"Compute the antiderivative of $3x^2$.", "integrate"),
        (r"Factor $x^2 - 4$.", "factor"),
        (r"Expand $(x+1)^3$.", "expand"),
        (r"Simplify $\frac{1}{2}$.", "simplify"),
        (r"Solve $x^2 - 5x + 6 = 0$.", "solve"),
        (r"Find the roots of $x^2 - 5x + 6$.", "solve"),
        (r"What is $24 \times 37$?", "numeric"),
        (r"Evaluate $\sqrt{17}$.", "numeric"),
        (r"Compute $1 + 1$.", "numeric"),
    ],
)
def test_the_problem_wording_selects_the_operation(problem: str, operation: str) -> None:
    assert compile_or_fail(problem).operation == operation


def test_the_more_specific_cue_wins_when_a_problem_mentions_several() -> None:
    """A problem that says "solve" and "factor" is a solve problem."""
    assert compile_or_fail(r"Solve by factoring: $x^2 - 4 = 0$.").operation == "solve"


@pytest.mark.parametrize(
    ("problem", "reason"),
    [
        # No math at all: a call would have to be invented.
        (r"A train travels 60 km in 2 hours. How fast?", "no_math_span"),
        (r"Which is larger, a dozen or a score?", "no_math_span"),
        # Constructs the tool's grammar cannot carry.
        (r"Evaluate $\log_{2} 8$.", "unconvertible_span"),
        (r"Evaluate $\lim_{x \to 0} \frac{\sin x}{x}$.", "unconvertible_span"),
        # An expression the tool can parse but the operation cannot use.
        (r"Find the derivative of $y = x^2 + 1$.", "unbuildable_diff"),
        (r"Find the integral of $y = x^2 + 1$.", "unbuildable_integrate"),
        # Two free symbols: no single variable to act on.
        (r"Solve $x + y = 3$.", "unbuildable_solve"),
        # ``numeric`` of a symbolic expression yields a symbol, not a number.
        (r"Evaluate $x^2 + 1$.", "unbuildable_numeric"),
    ],
)
def test_rejections_carry_a_stable_reason(problem: str, reason: str) -> None:
    compilation = compile_sympy_call(problem)

    assert compilation.arguments is None
    assert compilation.reason == reason


def test_a_problem_without_a_delimited_span_needs_an_explicit_cue() -> None:
    """Bare arithmetic is only read when the problem asks for a value."""
    cued = compile_or_fail(r"Evaluate 2 + 3.")
    assert cued.expression == "2+3"

    uncued = compile_sympy_call("There were 2 apples and then 3 more.")
    assert uncued.arguments is None


def test_candidates_are_ranked_by_how_much_computation_they_carry() -> None:
    """Incidental notation must not outrank the expression the question turns on."""
    candidates = compile_sympy_candidates(r"Compute $1$ and then $24 \times 37$.")

    expressions = [item.arguments.expression for item in candidates if item.arguments]
    assert expressions == ["24*37", "1"]


def test_an_equation_outranks_a_loose_expression() -> None:
    candidates = compile_sympy_candidates(r"Solve $3y = 9$ given that $x + 1$.")

    expressions = [item.arguments.expression for item in candidates if item.arguments]
    assert expressions == ["(3*y)-(9)", "x+1"]


def test_candidates_collapse_duplicates_and_honor_the_limit() -> None:
    problem = r"Compute $1 + 1$, and again $1 + 1$."

    candidates = compile_sympy_candidates(problem, limit=4)

    assert len(candidates) == 1
    assert candidates[0].arguments is not None
    assert candidates[0].arguments.expression == "1+1"


def test_the_single_call_entry_point_is_the_top_candidate() -> None:
    problem = r"Compute $1 + 1$ and $2 + 2$."

    assert compile_sympy_call(problem) == compile_sympy_candidates(problem, limit=1)[0]


def test_compilation_is_a_total_function_of_the_problem_text_alone() -> None:
    """No hidden reference reaches the compiler: same text, same call."""
    problem = r"Find the roots of $x^2 - 5x + 6 = 0$."

    first = compile_sympy_call(problem)
    second = compile_sympy_call(problem)

    assert first == second
    assert isinstance(first, Compilation)
