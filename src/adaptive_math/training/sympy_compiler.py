"""Deterministic compilation of a problem statement into one SymPy call.

This is the missing source of tool-use demonstrations.  The existing DIRECT
materializer renders an empty registry, so every record it produces teaches
"there are no tools"; the policy then never emits ``<tool_call>`` and GRPO has
no within-group variance to learn from (a constant reward term has no
gradient).  Producing demonstrations requires a call to demonstrate, and the
only trustworthy place to get one without a teacher model is the problem text
itself.

Two rules keep this honest:

* It reads the **problem only**.  The reference answer never reaches the
  compiler, so a compiled call cannot be a laundered label.
* It only *proposes*.  Nothing here decides correctness: the caller executes
  the call with the production ``SympyTool`` and keeps it only if the execution
  succeeds and the tool's own output verifies against the hidden reference.

The output is deliberately restricted to the grammar ``SympyTool._reject``
accepts (bare identifiers as symbols, ``sin``/``cos``/``tan``/``sqrt``/``pi``/``E``
as the only callables) so a compiled call is never rejected by the tool's own
safety boundary.
"""

import re
from dataclasses import dataclass

from adaptive_math.tools.sympy_tool import SympyArguments

# Delimited math, in the order LaTeX authors actually write it.  ``$$`` must be
# tried before ``$`` so an even-span problem does not yield a stray ``$`` body.
_MATH_SPANS = (
    re.compile(r"\$\$(?P<body>.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(?P<body>.+?)\\\]", re.DOTALL),
    re.compile(r"\\\((?P<body>.+?)\\\)", re.DOTALL),
    re.compile(r"\$(?P<body>[^$]+?)\$", re.DOTALL),
)

# A bare arithmetic run, used only when the problem carries an explicit
# evaluate cue and no delimited span survived.  Bounded on both sides so prose
# like "the 3 remaining 4 items" cannot masquerade as an expression.
_BARE_ARITHMETIC = re.compile(
    r"(?<![\w$.])(?P<body>\d+(?:\.\d+)?(?:\s*[-+*/^]\s*\d+(?:\.\d+)?)+)(?![\w$])"
)

_EVALUATE_CUE = re.compile(
    r"\b(?:evaluate|compute|calculate|what\s+is|find\s+the\s+value|simplify|"
    r"expand|factor|derivative|integral|solve)\b",
    re.IGNORECASE,
)

# First match wins.  Ordering matters: a problem that says "solve" and also
# mentions "factor" is a solve problem, so the more specific cues come first.
_OPERATION_CUES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:derivative|differentiate|d/dx)\b", re.IGNORECASE), "diff"),
    (re.compile(r"\b(?:integral|integrate|antiderivative)\b", re.IGNORECASE), "integrate"),
    (re.compile(r"\b(?:factor|factorize|factorise)\b", re.IGNORECASE), "factor"),
    (re.compile(r"\bexpand\b", re.IGNORECASE), "expand"),
    (re.compile(r"\bsimplify\b", re.IGNORECASE), "simplify"),
    (
        re.compile(
            r"\b(?:solve|solutions?|roots?|zeros?|equations?)\b", re.IGNORECASE
        ),
        "solve",
    ),
)

# Macros with a fixed meaning.  Anything longer than one character is handled
# before braces are stripped so ``\pi`` cannot decay into a bare identifier.
_SIMPLE_MACROS: tuple[tuple[str, str], ...] = (
    (r"\cdot", "*"),
    (r"\times", "*"),
    (r"\div", "/"),
    (r"\pi", "pi"),
    (r"\left", ""),
    (r"\right", ""),
    (r"\{,}", ""),  # LaTeX thousands separator: 1{,}000 -> 1000
    (r"\,", ""),
    (r"\;", ""),
    (r"\:", ""),
    (r"\!", ""),
    (r"\ ", " "),
)

# Macros whose meaning cannot be carried by a single expression.  Rejecting is
# better than silently dropping a constraint and compiling the wrong call.
_UNSUPPORTED = re.compile(
    r"\\(?:pm|mp|infty|leq|geq|neq|approx|equiv|in|notin|subset|cup|cap|"
    r"sum|prod|lim|log|ln|exp|sin|cos|tan|alpha|beta|theta|deg|circ|%|"
    r"begin|end|text|mathrm|operatorname|boxed|overline|hat|vec)"
)

_ALLOWED_CALLABLES = frozenset({"sin", "cos", "tan", "sqrt"})
_ALLOWED_CONSTANTS = frozenset({"pi", "E"})
_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
# A bare identifier directly before ``(`` is juxtaposition, not a call: the
# tool treats only its own allowlisted callables as functions.
_IMPLICIT_BEFORE_GROUP = re.compile(r"(?<![\w.])([A-Za-z_]\w*)(?=\()")
# Everything ``sympy.sympify`` accepts here, and nothing else.  Passing
# ``SympyTool._reject`` only proves an expression is safe; it says nothing
# about whether the parser can read it, and ``(3;3)``, ``0<x<1`` and ``|x|``
# all sail through the safety boundary and then fail at execution.  ``=`` is
# legal because ``solve`` equations carry one until ``_build`` rewrites them.
_ALLOWED_EXPRESSION = re.compile(r"^[A-Za-z0-9_+\-*/^()=.]+$")
# An operator with nothing to operate on: ``(x^7)-()``, ``2+``, ``=1:2``.  A
# leading ``-`` is a sign, not an empty operand, so it stays allowed.
_EMPTY_OPERAND = re.compile(r"\(\)|[+\-*/^]\s*\)|[+\-*/^]\s*$|^\s*[^A-Za-z0-9_(-]")
# Mirrors SympyTool._reject: characters that could escape the expression parser.
_FORBIDDEN_SUBSTRINGS = ("__", ".", "[", "]", "{", "}", "'", '"', "import")


@dataclass(frozen=True)
class Compilation:
    """One proposed call plus why it was or was not produced.

    ``reason`` is ``"ok"`` on success and a stable, low-cardinality tag
    otherwise, so callers can count rejections by cause without parsing prose.
    """

    arguments: SympyArguments | None
    reason: str


def compile_sympy_call(problem: str) -> Compilation:
    """Propose the single most promising SymPy call, or why none fits."""
    return compile_sympy_candidates(problem, limit=1)[0]


def compile_sympy_candidates(problem: str, limit: int = 4) -> tuple[Compilation, ...]:
    """Every distinct call the problem text supports, most promising first.

    A word problem carries incidental math (``$1$`` in "prove that the number
    $1$ can be written...") as well as the expression the question turns on.
    The first span in the text is therefore often not the one worth computing,
    so the caller gets the whole ranked list and tries as many as it is willing
    to pay for.  Only a real execution plus the verifier can accept one.

    The order is a function of the problem text alone: the reference answer is
    never an input, so a chosen candidate cannot be a laundered label.  The
    failure reasons are reported once, from the highest-ranked span that
    failed, so callers can still count rejections by cause.
    """
    operation = _select_operation(problem)
    spans = _math_spans(problem)
    if not spans and _EVALUATE_CUE.search(problem):
        spans = [match.group("body") for match in _BARE_ARITHMETIC.finditer(problem)]
    if not spans:
        return (Compilation(None, "no_math_span"),)

    ranked: list[tuple[int, int, Compilation]] = []
    saw_expression = False
    for index, span in enumerate(spans):
        expression = _to_sympy(span)
        if expression is None:
            continue
        saw_expression = True
        arguments = _build(operation, expression)
        if arguments is None:
            continue
        ranked.append((-_span_score(expression), index, Compilation(arguments, "ok")))
    if not ranked:
        if saw_expression:
            return (Compilation(None, f"unbuildable_{operation}"),)
        return (Compilation(None, "unconvertible_span"),)

    ranked.sort(key=lambda item: (item[0], item[1]))
    distinct: list[Compilation] = []
    seen: set[tuple[str, str]] = set()
    for _, _, compilation in ranked:
        assert compilation.arguments is not None
        key = (compilation.arguments.operation, compilation.arguments.expression)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(compilation)
        if len(distinct) == limit:
            break
    return tuple(distinct)


# Text-only signals that a span is the computation the question turns on
# rather than incidental notation.  An equation is the strongest: problems
# rarely state one they do not intend to be solved.
def _span_score(expression: str) -> int:
    score = 4 if "=" in expression else 0
    score += min(sum(expression.count(operator) for operator in "+-*/^"), 4)
    if _IDENTIFIER.search(expression) and not _is_callable_only(expression):
        score += 2
    return score + min(len(expression) // 8, 3)


def _select_operation(problem: str) -> str:
    for pattern, operation in _OPERATION_CUES:
        if pattern.search(problem):
            return operation
    return "numeric"


def _math_spans(problem: str) -> list[str]:
    """Delimited math bodies in document order, de-duplicated."""
    found: list[tuple[int, str]] = []
    for pattern in _MATH_SPANS:
        for match in pattern.finditer(problem):
            found.append((match.start(), match.group("body")))
    found.sort(key=lambda item: item[0])
    seen: set[str] = set()
    spans: list[str] = []
    for _, body in found:
        text = body.strip()
        if text and text not in seen:
            seen.add(text)
            spans.append(text)
    return spans


def _to_sympy(span: str) -> str | None:
    """Translate one LaTeX span into the SympyTool expression grammar."""
    text = _expand_braced_macros(span.strip())
    if text is None:
        return None
    for macro, replacement in _SIMPLE_MACROS:
        text = text.replace(macro, replacement)
    if _UNSUPPORTED.search(text):
        return None
    # Remaining braces only group: ``x^{2}`` -> ``x^2``.
    text = text.replace("{", "").replace("}", "")
    if "\\" in text:
        return None
    text = re.sub(r"\s+", "", text)
    text = _insert_multiplication(text)
    if not text or not re.search(r"\d", text):
        return None
    if any(token in text for token in _FORBIDDEN_SUBSTRINGS):
        return None
    if text.count("(") != text.count(")"):
        return None
    # ``_reject`` proves an expression is *safe*; it does not prove the parser
    # can read it.  These two checks close that gap for every construct the
    # corpus actually produced (``|x|``, ``0<x<1``, ``(3;3)``, ``(x^7)-()``).
    if _ALLOWED_EXPRESSION.fullmatch(text) is None:
        return None
    if _EMPTY_OPERAND.search(text):
        return None
    return text


def _insert_multiplication(text: str) -> str:
    """Make juxtaposed factors explicit, the only form ``sympify`` parses.

    LaTeX multiplies by juxtaposition (``3x``, ``2(x+1)``, ``(x+1)(x-1)``) and
    ``sympy.sympify`` — the tool's parser — has no implicit-multiplication
    transformation: it raises ``SympifyError`` on ``3x``.  A compiled call that
    keeps the LaTeX form would be a guaranteed execution failure.

    A bare symbol followed by ``(`` becomes a product too.  Function
    application the tool cannot perform (``f(x)``) then dies downstream at
    variable extraction, which sees two free symbols instead of one, rather
    than silently returning the wrong number.
    """
    text = re.sub(r"(?<=[\d)])(?=[A-Za-z_(])", "*", text)
    return _IMPLICIT_BEFORE_GROUP.sub(
        lambda match: (
            match.group(1)
            if match.group(1) in _ALLOWED_CALLABLES
            else f"{match.group(1)}*"
        ),
        text,
    )


def _expand_braced_macros(span: str) -> str | None:
    """Rewrite ``\\frac``/``\\sqrt`` innermost-first using brace matching.

    A regex cannot parse these: the arguments nest (``\\frac{\\sqrt{2}}{2}``),
    and a non-greedy pattern silently mis-pairs the braces instead of failing.
    """
    text = span
    while True:
        match = _find_innermost_braced_macro(text)
        if match is None:
            break
        start, end, name, argument = match
        if name == "frac":
            # ``argument`` is already the numerator; ``end`` sits on the
            # denominator's opening brace.  Re-reading the numerator from
            # ``end`` would consume the denominator instead.
            numerator = argument
            group = _take_group(text, end)
            if group is None:
                return None
            denominator, after = group
            text = f"{text[:start]}(({numerator})/({denominator})){text[after:]}"
        else:
            replacement = f"sqrt({argument})"
            text = f"{text[:start]}{replacement}{text[end:]}"
    return text


def _find_innermost_braced_macro(text: str) -> tuple[int, int, str, str] | None:
    """Locate the last ``\\frac``/``\\sqrt`` whose argument braces are balanced.

    The last balanced match is the innermost one: any macro nested inside an
    earlier match's body necessarily starts after it, so it would be matched
    later still.  Returning the first match instead mis-pairs the braces of a
    nested ``\\frac`` — the denominator gets consumed as the numerator.

    ``end`` is the index of the opening brace of the *next* group, which is
    what ``_expand_braced_macros`` needs to find a ``\\frac`` denominator.
    """
    found: tuple[int, int, str, str] | None = None
    for match in re.finditer(r"\\(frac|sqrt)\{", text):
        # ``match.end()`` sits *after* the opening brace; ``_take_group`` wants
        # the index *of* it.  An off-by-one here makes every macro look
        # unbalanced and silently disables the whole expansion.
        group = _take_group(text, match.end() - 1)
        if group is None:
            continue
        argument, after = group
        found = (match.start(), after, match.group(1), argument)
    return found


def _take_group(text: str, open_index: int) -> tuple[str, int] | None:
    """Return the balanced ``{...}`` body starting at ``open_index``, and its end."""
    if open_index >= len(text) or text[open_index] != "{":
        return None
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index], index + 1
    return None


def _build(operation: str, expression: str) -> SympyArguments | None:
    """Assemble a validated call, deriving the variable when the operation needs one."""
    if "=" in expression and operation != "solve":
        # Only ``solve`` consumes an equation, and it rewrites the two sides
        # into one expression below.  Every other operation takes an
        # expression: an equation has no derivative to take, nothing to
        # factor, and ``expand(f(z)=...)`` is a function definition, not math
        # the tool can evaluate.
        return None
    if operation in {"solve", "diff", "integrate"}:
        variable = _solve_variable(expression)
        if variable is None:
            return None
        if "=" in expression:
            # ``solve`` wants an expression, not an equation: ``x^2-5x+6=0``
            # is reduced to ``(x^2-5x+6)-(0)`` so the tool never has to
            # understand ``=`` and the roots stay the roots.
            left, _, right = expression.partition("=")
            if not left or not right:
                return None
            expression = f"({left})-({right})"
        payload: dict[str, object] = {
            "operation": operation,
            "expression": expression,
            "variables": [variable],
        }
    elif operation == "numeric":
        # ``numeric`` of a free-symbol expression yields a symbolic result, not
        # a number, so it cannot support a numeric final answer.  The other
        # symbolic operations are *expected* to take free symbols.
        if _IDENTIFIER.search(expression) and not _is_callable_only(expression):
            return None
        payload = {"operation": operation, "expression": expression, "variables": []}
    else:
        payload = {"operation": operation, "expression": expression, "variables": []}
    try:
        return SympyArguments.model_validate(payload)
    except ValueError:
        return None


def _solve_variable(expression: str) -> str | None:
    """The single free symbol, or ``None`` when the expression is ambiguous."""
    symbols: set[str] = {
        name
        for name in _IDENTIFIER.findall(expression)
        if name not in _ALLOWED_CALLABLES and name not in _ALLOWED_CONSTANTS
    }
    if len(symbols) != 1:
        return None
    return next(iter(symbols))


def _is_callable_only(expression: str) -> bool:
    """True when every identifier is an allowlisted callable or constant."""
    identifiers = set(_IDENTIFIER.findall(expression)) - {"E"}
    return identifiers <= (_ALLOWED_CALLABLES | _ALLOWED_CONSTANTS)
