from adaptive_math.verifier.symbolic import compare_expressions


def compare(prediction: str, reference: str, task_id: str = "test") -> dict[str, object]:
    return compare_expressions(prediction, reference, task_id)


def test_x_plus_x_equals_2x() -> None:
    assert compare("x+x", "2*x")["status"] == "correct"


def test_rational_domains_are_preserved() -> None:
    # (x^2-1)/(x-1) does not globally equal x+1 because the domains differ
    assert compare(r"\frac{x^2-1}{x-1}", "x+1")["status"] == "incorrect"


def test_pythagorean_identity() -> None:
    assert compare(r"\sin(x)^2+\cos(x)^2", "1")["status"] == "correct"


def test_free_symbol_mismatch_is_incorrect() -> None:
    assert compare("x+1", "y+1")["status"] == "incorrect"


def test_malformed_latex_is_invalid() -> None:
    assert compare(r"\frac{1}{", r"\frac{1}{2}")["status"] == "invalid_prediction"
    assert compare(r"\frac{1}{2}", r"\frac{1}{")["status"] == "invalid_reference"


def test_numeric_constants_compare() -> None:
    assert compare("3", "3.0")["status"] == "correct"
    assert compare("3", "4")["status"] == "incorrect"


def test_unknown_functions_are_invalid() -> None:
    assert compare("f(x)", "f(x)")["status"] == "invalid_prediction"


def test_differing_polynomials_are_incorrect() -> None:
    assert compare("x^2 + 1", "x^2 + 2")["status"] == "incorrect"


def test_correct_verdicts_have_reward_one() -> None:
    assert compare("x+x", "2*x")["reward"] == 1.0
    assert compare("x+1", "x+2")["reward"] == 0.0


def test_verification_is_deterministic_for_the_same_task() -> None:
    first = compare("x^3 - x", "x*(x^2-1)")
    second = compare("x^3 - x", "x*(x^2-1)")
    assert first == second


def test_numeric_cross_check_records_method_and_points() -> None:
    """Retargeted 2026-09-25: it used x^2+1 vs x^2+2, whose difference reduces to
    the constant -1, and such pairs are now decided exactly -- a proved non-zero
    difference is better evidence than five sampled points. x^2 vs x^3 still has to
    go through sampling, so the requirement this test guards is still guarded."""
    verdict = compare("x^2", "x^3")
    assert verdict["status"] == "incorrect"
    assert "numeric" in str(verdict["details"].get("method"))
    assert "points" in verdict["details"]


def test_a_constant_difference_is_decided_without_sampling() -> None:
    verdict = compare("x^2 + 1", "x^2 + 2")

    assert verdict["status"] == "incorrect"
    assert verdict["details"]["method"] == "exact"
