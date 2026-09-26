"""Symbolic expression comparison.

This module is designed to run ONLY inside the isolated verifier worker
process (verifier.worker): it parses untrusted LaTeX with math-verify,
compares with SymPy and cross-checks numerically at deterministic points.
It never imports model, agent or network code and never executes input.
"""

import hashlib
import math
import random
import re

import sympy
from math_verify.parser import parse_latex_cached
from sympy.core.relational import Relational as _Relational

from adaptive_math.verifier.normalizer import normalize_surface

# A "No" reference and a "No" prediction are words, not products of symbols.
_WORD_ANSWER = re.compile(r"^(yes|no|true|false)$", re.IGNORECASE)
_TEXT_WRAPPER = re.compile(
    r"\\(?:text|textrm|textnormal|mathrm|mathsf|mathit|mathbf|operatorname)\s*\{([^{}]*)\}"
)
_OPEN_ENDED = re.compile(r"\\cdots\b|\\ldots\b|\\vdots\b|\u2026")

MAX_EXPR_CHARS = 8192
MAX_NODE_COUNT = 4096
MAX_FREE_SYMBOLS = 32
REQUIRED_POINTS = 5
MAX_POINT_ATTEMPTS = 200
POINT_RANGE = 5.0
CROSS_ATOL = 1e-6
CROSS_RTOL = 1e-6

_APPLIED_UNDEF = sympy.core.function.AppliedUndef


def compare_expressions(prediction: str, reference: str, task_id: str) -> dict[str, object]:
    """Compare two LaTeX expressions and return a VerifierResult-shaped dict.

    Never raises. Order of evidence: canonical equality, domain-aware SymPy
    simplify difference, then a numeric cross-check at deterministic
    non-singular points seeded from task_id. The cross-check only returns
    correct when both expressions are defined at all points with error below
    tolerance; if expressions cannot be evaluated at all, the prediction is
    unverifiable (invalid) rather than guessed incorrect.
    """
    pred_norm = normalize_surface(prediction)
    ref_norm = normalize_surface(reference)
    if len(pred_norm) > MAX_EXPR_CHARS:
        return _verdict("invalid_prediction", reason="expression exceeds length limit")
    if len(ref_norm) > MAX_EXPR_CHARS:
        return _verdict("invalid_reference", reason="expression exceeds length limit")
    # Words first: "No" parses as the product n*o and "\text{No}" as something
    # else again, so a correct yes/no answer used to come back *incorrect* with
    # "free symbol mismatch". Yes/no problems are common in this pool.
    pred_word = _as_word(pred_norm)
    ref_word = _as_word(ref_norm)
    if pred_word is not None or ref_word is not None:
        if pred_word is not None and ref_word is not None:
            return _verdict(
                "correct" if pred_word == ref_word else "incorrect",
                normalized_prediction=pred_word,
                normalized_reference=ref_word,
                details={"method": "word"},
            )
        # One side says yes/no and the other is a value or a structure. That is a
        # wrong answer, not an unreadable one: the audit rows that answered `a=b`
        # and a group decomposition to a "No" reference are simply incorrect.
        return _verdict(
            "incorrect",
            normalized_prediction=pred_norm,
            normalized_reference=ref_norm,
            details={"method": "word", "reason": "word_vs_value"},
        )
    pred_open = _OPEN_ENDED.search(pred_norm) is not None
    ref_open = _OPEN_ENDED.search(ref_norm) is not None
    if pred_open and ref_open:
        # Two unfinished claims: no honest comparison exists.
        return _verdict("invalid_prediction", reason="open_ended_series")
    if pred_open:
        # A trailing \cdots cannot equal a closed form: a wrong answer, not an
        # unreadable one. Reward is 0.0 either way, so this cannot false-accept.
        return _verdict(
            "incorrect",
            normalized_prediction=pred_norm,
            normalized_reference=ref_norm,
            details={"method": "open_series", "reason": "open_ended_series"},
        )
    if ref_open:
        # An open-ended reference against a clean prediction is a broken reference.
        return _verdict("invalid_reference", reason="open_ended_series")
    pred_sym = _parse_to_sympy(pred_norm)
    if pred_sym is None:
        return _verdict("invalid_prediction", reason="malformed expression")
    ref_sym = _parse_to_sympy(ref_norm)
    if ref_sym is None:
        return _verdict("invalid_reference", reason="malformed expression")
    if _node_count(pred_sym) > MAX_NODE_COUNT:
        return _verdict("invalid_prediction", reason="expression has too many nodes")
    if _node_count(ref_sym) > MAX_NODE_COUNT:
        return _verdict("invalid_reference", reason="expression has too many nodes")
    if len(pred_sym.free_symbols) > MAX_FREE_SYMBOLS:
        return _verdict("invalid_prediction", reason="too many free symbols")
    if len(ref_sym.free_symbols) > MAX_FREE_SYMBOLS:
        return _verdict("invalid_reference", reason="too many free symbols")
    if pred_sym.atoms(_APPLIED_UNDEF) or ref_sym.atoms(_APPLIED_UNDEF):
        # unknown functions like f(x) cannot be verified reliably
        if pred_sym.atoms(_APPLIED_UNDEF):
            return _verdict("invalid_prediction", reason="unknown function application")
        return _verdict("invalid_reference", reason="unknown function application")

    pred_canon = str(pred_sym)
    ref_canon = str(ref_sym)
    if pred_canon == ref_canon:
        return _verdict(
            "correct",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "canonical"},
        )

    if not isinstance(pred_sym, sympy.Expr) or not isinstance(ref_sym, sympy.Expr):
        # Relations, sets and tuples cannot be subtracted; the arithmetic path
        # below would raise inside its own exception handler.
        return _compare_structural(pred_sym, ref_sym, pred_canon, ref_canon)

    try:
        diff = sympy.simplify(pred_sym - ref_sym)
    except Exception:
        diff = pred_sym - ref_sym
    if diff == 0:
        if _domains_agree(pred_sym, ref_sym):
            return _verdict(
                "correct",
                normalized_prediction=pred_canon,
                normalized_reference=ref_canon,
                details={"method": "simplify"},
            )
        return _verdict(
            "incorrect",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "simplify", "reason": "domains differ"},
        )

    if isinstance(diff, (sympy.Integer, sympy.Rational)):
        # sympy actually computed the difference and it is a concrete rational, so
        # it is proved non-zero and no floating-point evidence can improve on that:
        # float() rounded 2**1000 and 2**1000 + 1 onto the same 1.07e301 and called
        # them equal, and it overflowed a pair of 1212-digit binomial coefficients
        # to inf, which came back as "could not be evaluated".
        #
        # The isinstance test is deliberately structural rather than
        # ``pred_sym.is_rational``: that assumption *evaluates* the expression, and
        # for a (5!)!-style factorial tower evaluation raises OverflowError out of a
        # function whose contract is to never raise.
        return _verdict(
            "incorrect",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "exact", "reason": "rational difference"},
        )

    if pred_sym.free_symbols != ref_sym.free_symbols:
        return _verdict(
            "incorrect",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "structure", "reason": "free symbol mismatch"},
        )
    return _numeric_cross_check(pred_sym, ref_sym, pred_canon, ref_canon, task_id)


def _as_word(text: str) -> str | None:
    """yes/no/true/false, with or without a \\text wrapper, else None."""
    unwrapped = _TEXT_WRAPPER.sub(lambda match: match.group(1), text)
    candidate = unwrapped.strip().rstrip(".!").replace(" ", "")
    match = _WORD_ANSWER.match(candidate)
    return match.group(1).lower() if match else None


def _relation_signature(value: sympy.Basic) -> tuple[str, str] | None:
    """(kind, difference) for a relational, normalised to one orientation.

    ``2 \\leq x`` and ``x \\geq 2`` are the same answer, so every form is read as
    "something <kind> 0" with kind in {gt, ge, eq, ne}. Strictness is part of the
    signature: ``x > 2`` must not match a reference of ``x \\geq 2``.
    """
    if not isinstance(value, _Relational):
        return None
    lhs, rhs = value.args
    difference = sympy.simplify(lhs - rhs)
    if isinstance(value, sympy.Equality):
        kind = "eq"
    elif isinstance(value, sympy.Unequality):
        kind = "ne"
    elif isinstance(value, (sympy.StrictGreaterThan, sympy.StrictLessThan)):
        kind = "gt"
    else:
        kind = "ge"
    if kind in {"gt", "ge"} and difference.could_extract_minus_sign():
        difference = -difference
    return kind, str(difference)


def _elements(value: sympy.Basic) -> list[sympy.Basic] | None:
    """Members of a set or tuple answer, or None if this is not one."""
    if isinstance(value, sympy.Tuple):
        return list(value.args)
    if isinstance(value, (sympy.FiniteSet, sympy.Range)):
        return list(value.args) if isinstance(value, sympy.FiniteSet) else None
    return None


def _compare_structural(
    pred_sym: sympy.Basic, ref_sym: sympy.Basic, pred_canon: str, ref_canon: str
) -> dict[str, object]:
    """Decide a pair where at least one side is not an arithmetic expression.

    Never guesses equality it cannot prove: two boolean combinations that differ
    come back invalid rather than "incorrect", because And/Or ordering is not a
    semantic difference. A relation against a bare value is a wrong answer,
    though -- those cannot denote the same thing.
    """
    pred_relation = _relation_signature(pred_sym)
    ref_relation = _relation_signature(ref_sym)
    if pred_relation is not None and ref_relation is not None:
        return _verdict(
            "correct" if pred_relation == ref_relation else "incorrect",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "relation", "signature": list(pred_relation)},
        )
    pred_elements = _elements(pred_sym)
    ref_elements = _elements(ref_sym)
    if pred_elements is not None and ref_elements is not None:
        agree = len(pred_elements) == len(ref_elements) and sorted(
            str(sympy.simplify(item)) for item in pred_elements
        ) == sorted(str(sympy.simplify(item)) for item in ref_elements)
        return _verdict(
            "correct" if agree else "incorrect",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "collection", "count": len(pred_elements)},
        )
    if pred_relation is not None or ref_relation is not None:
        return _verdict(
            "incorrect",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "structure", "reason": "relation_vs_value"},
        )
    if pred_elements is not None or ref_elements is not None:
        return _verdict(
            "incorrect",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={"method": "structure", "reason": "collection_vs_value"},
        )
    return _verdict(
        "invalid_prediction",
        normalized_prediction=pred_canon,
        normalized_reference=ref_canon,
        details={"method": "structure", "reason": "unsupported_form"},
    )


def _parse_to_sympy(text: str) -> sympy.Basic | None:
    """Parse via math-verify's LaTeX-to-sympy layer (latex2sympy2).

    math-verify 0.9's high-level parse() extracts numeric answers from
    solution text and cannot handle symbolic expressions; the parsing layer
    underneath it (parse_latex_cached) is the right entry point for
    already-normalized expressions.
    """
    try:
        parsed = parse_latex_cached(text)
    except Exception:
        return None
    # Anything the parser can express as a sympy node is a candidate: an Expr for
    # a value, a Relational for an inequality, a FiniteSet/Tuple for a
    # multi-element answer. The earlier Expr-only gate read all three of those as
    # "malformed expression", which is where four of the six audit formats died.
    return parsed if isinstance(parsed, sympy.Basic) else None


def _node_count(expr: sympy.Expr) -> int:
    count = 0
    for _ in sympy.preorder_traversal(expr):
        count += 1
        if count > MAX_NODE_COUNT:
            break
    return count


def _domains_agree(a: sympy.Expr, b: sympy.Expr) -> bool:
    """Compare singularities over every shared free symbol. A simplify
    difference of zero only proves equality where both sides are defined."""
    for symbol in sorted(a.free_symbols | b.free_symbols, key=str):
        try:
            if sympy.singularities(a, symbol) != sympy.singularities(b, symbol):
                return False
        except (NotImplementedError, TypeError, AttributeError):
            continue  # undecidable here; rely on the simplify evidence
    return True


def _numeric_cross_check(
    pred_sym: sympy.Expr,
    ref_sym: sympy.Expr,
    pred_canon: str,
    ref_canon: str,
    task_id: str,
) -> dict[str, object]:
    symbols = sorted(pred_sym.free_symbols, key=str)
    seed = hashlib.sha256(f"{task_id}\x00{pred_canon}\x00{ref_canon}".encode()).digest()
    rng = random.Random(seed)
    points: list[tuple[dict[sympy.Symbol, float], float]] = []
    attempts = 0
    while len(points) < REQUIRED_POINTS and attempts < MAX_POINT_ATTEMPTS:
        attempts += 1
        subs = {symbol: rng.uniform(-POINT_RANGE, POINT_RANGE) for symbol in symbols}
        try:
            p_val = _evaluate(pred_sym, subs)
            r_val = _evaluate(ref_sym, subs)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            continue  # singular or unrepresentable point; try the next one
        if p_val is None or r_val is None:
            continue
        error = abs(p_val - r_val)
        if not math.isfinite(error):
            continue
        points.append((subs, error))
    recorded_points: dict[str, object] = {
        str(symbol): [round(subs[symbol], 6) for subs, _ in points] for symbol in symbols
    }
    if len(points) < REQUIRED_POINTS:
        return _verdict(
            "invalid_prediction",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={
                "method": "numeric",
                "reason": "expressions could not be evaluated at sample points",
                "points": recorded_points,
            },
        )
    max_error = max(error for _, error in points)
    max_ref = max(abs(_evaluate(ref_sym, subs) or 0.0) for subs, _ in points)
    tolerance = CROSS_ATOL + CROSS_RTOL * max_ref
    if max_error <= tolerance:
        return _verdict(
            "correct",
            normalized_prediction=pred_canon,
            normalized_reference=ref_canon,
            details={
                "method": "numeric",
                "points": recorded_points,
                "max_error": max_error,
            },
        )
    return _verdict(
        "incorrect",
        normalized_prediction=pred_canon,
        normalized_reference=ref_canon,
        details={
            "method": "numeric",
            "points": recorded_points,
            "max_error": max_error,
        },
    )


def _evaluate(expr: sympy.Expr, subs: dict[sympy.Symbol, float]) -> complex | None:
    value = sympy.N(expr.subs(subs), 15)
    try:
        re_part, im_part = value.as_real_imag()
        result = complex(float(re_part), float(im_part))
    except (TypeError, ValueError, AttributeError):
        return None
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        return None
    return result


def _verdict(
    status: str,
    *,
    reason: str | None = None,
    normalized_prediction: str | None = None,
    normalized_reference: str | None = None,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    merged = dict(details or {})
    if reason:
        merged["reason"] = reason
    return {
        "status": status,
        "reward": 1.0 if status == "correct" else 0.0,
        "normalized_prediction": normalized_prediction,
        "normalized_reference": normalized_reference,
        "details": merged,
    }
