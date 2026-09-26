"""Verifier coverage for the six answer formats named in the 2026-09-19 invalid audit.

Source rows: artifacts/audit/invalid-classification-20260919/classification.jsonl
(9 rows across 3 arms, all with extract_status=ok). Every one of them was checked
against its reference and 0 of 9 were actually correct, so the immediate cost is
label quality -- but the reason these are promotion blockers is the other half of
each class: a RIGHT answer written in one of these formats currently earns zero
reward in GRPO. Each class therefore carries both a wrong-in-this-format case,
which must be judged *incorrect*, and a right-in-this-format case, which must be
judged *correct*. "Cannot read it" must never be silently equal to "no credit".
"""

from adaptive_math.verifier.symbolic import compare_expressions


def compare(prediction: str, reference: str, task_id: str = "gap-probe") -> dict[str, object]:
    return compare_expressions(prediction, reference, task_id)


# --- A. yes/no answers, the highest-impact gap ----------------------------------
# Before this file, verify_answer("No", "\\text { No }") returned *incorrect*:
# math-verify parses \text{No} as the product n*o, so every correct yes/no answer
# in an OmniMath-style pool lost its reward.


def test_a_word_answer_matches_a_textual_reference() -> None:
    assert compare("No", r"\text { No }")["status"] == "correct"
    assert compare("Yes", r"\text{Yes}")["status"] == "correct"


def test_a_word_answer_is_not_over_accepted() -> None:
    assert compare("No", r"\text{Yes}")["status"] == "incorrect"
    assert compare("Yes", "No")["status"] == "incorrect"


def test_an_equation_answer_against_a_yes_no_reference_is_incorrect() -> None:
    """omni_math:0adb65d7 -- the model answered `a=b` where the reference is No."""
    assert compare("a=b", r"\text { No }")["status"] == "incorrect"


def test_a_structure_answer_against_a_yes_no_reference_is_incorrect() -> None:
    """omni_math:0986e00a -- a group decomposition where the reference is No."""
    prediction = r"\mathbb{Z}_{2^{1005}}\times \mathbb{Z}_{2^{1005}}\times \cdots"
    assert compare(prediction, "No")["status"] == "incorrect"


# --- B. relations ---------------------------------------------------------------


def test_identical_relations_are_correct() -> None:
    """A relational answer is not a sympy.Expr, and the parse gate used to reject
    everything that was not one: `x \\geq 2` against itself came back invalid."""
    assert compare(r"x \geq 2", r"x \geq 2")["status"] == "correct"


def test_a_relation_with_reversed_operands_is_still_correct() -> None:
    assert compare(r"2 \leq x", r"x \geq 2")["status"] == "correct"


def test_strict_and_non_strict_relations_are_not_conflated() -> None:
    assert compare(r"x > 2", r"x \geq 2")["status"] == "incorrect"


def test_a_relation_against_a_numeric_reference_is_incorrect() -> None:
    """omni_math:0d9e9393 -- `b^2 - 4ac \\geq 0` where the area is 49\\pi."""
    assert compare(r"b^2 - 4ac \geq 0", r"49 \pi")["status"] == "incorrect"


# --- C. angle spelling and multi-value answers ----------------------------------


def test_degree_macro_equals_circ_superscript() -> None:
    assert compare(r"60\degree", r"60^\circ")["status"] == "correct"


def test_a_triple_of_angles_against_one_angle_is_incorrect() -> None:
    """omni_math:0a28be89 -- `90\\degree ,60\\degree ,30\\degree` vs 60^\\circ."""
    assert compare(r"90\degree ,60\degree ,30\degree", r"60^\circ")["status"] == "incorrect"


# --- D. huge exact constants ----------------------------------------------------


def test_two_unevaluated_huge_binomials_are_compared_exactly() -> None:
    """omni_math:0a6a00a1. Both sides evaluate to ~1200-digit integers; the float
    conversion in the numeric cross-check overflowed to inf, every sample point was
    discarded, and the verdict was "could not be evaluated"."""
    assert compare(r"\binom{4032}{2016}", r"\binom{4030}{2015}")["status"] == "incorrect"


def test_an_identical_huge_binomial_is_correct() -> None:
    assert compare(r"\binom{4030}{2015}", r"\binom{4030}{2015}")["status"] == "correct"


def test_a_huge_power_off_by_one_is_incorrect() -> None:
    assert compare("2^{1000}", "2^{1000}+1")["status"] == "incorrect"


# --- E. factorial power syntax --------------------------------------------------


def test_factorial_power_syntax_is_parseable_and_wrong_answers_fail() -> None:
    """omni_math:0c22491e -- 8!/(2!^4) is 2520, the reference is 105."""
    assert compare(r"8!/(2!^4)", "105")["status"] == "incorrect"


def test_factorial_power_syntax_used_correctly_earns_credit() -> None:
    """The reward-eating half of the same format: 8!/((2!)^4 4!) is exactly 105."""
    assert compare(r"8!/(2!^4 \cdot 4!)", "105")["status"] == "correct"


# --- F. open-ended series: label honestly, do not guess -------------------------


def test_an_open_ended_prediction_against_a_closed_reference_is_incorrect() -> None:
    """omni_math:02f104ee. A trailing ``\\cdots`` claims an unfinished series;
    against a closed form it cannot be equal, so it is a wrong answer, not an
    unreadable one."""
    prediction = r"\frac{1}{\sqrt{5}} + \frac{2}{5} + \frac{3}{25} + \cdots"
    verdict = compare(prediction, r"(2i-1)/4")

    assert verdict["status"] == "incorrect"
    assert verdict["details"]["method"] == "open_series"
    assert verdict["details"]["reason"] == "open_ended_series"


def test_a_closed_prediction_against_an_open_ended_reference_is_an_invalid_reference() -> None:
    verdict = compare(r"1/2", r"\sum_{k=1}^{n} k + \cdots")

    assert verdict["status"] == "invalid_reference"
    assert verdict["details"]["reason"] == "open_ended_series"


def test_two_open_ended_expressions_stay_unverifiable() -> None:
    verdict = compare(r"1 + 2 + \cdots", r"3 + \cdots")

    assert verdict["status"] == "invalid_prediction"
    assert verdict["details"]["reason"] == "open_ended_series"


def test_a_difference_sympy_cannot_evaluate_is_not_called_incorrect() -> None:
    """A regression this file's own fixes introduced. ``simplify(pred - ref) == 0``
    is a syntactic test: for ``(5!)!``-style towers sympy leaves the factorial
    unevaluated, reports a non-zero difference it never computed, and an
    exact-rational shortcut built on that would grade a possibly-correct answer as
    wrong. Unprovable stays unverifiable -- the adversarial fixture already demands
    it, and the reward must not invent certainty the verifier does not have."""
    verdict = compare("(((((5!)!)!)!)!)", "120")

    assert verdict["status"] in {"invalid_prediction", "timeout"}
