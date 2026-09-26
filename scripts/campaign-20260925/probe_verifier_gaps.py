"""Probe the six verifier coverage gaps named in the 2026-09-19 invalid audit.

For each gap class we ask two questions:

  negative case -- the row the audit found (wrong answer in an awkward format).
      Today it comes back invalid_prediction; the fix must make it *incorrect*,
      because "the model answered in a form we cannot read" and "the model gave a
      wrong answer" are different facts for the failure miner and for labels.

  positive case -- a RIGHT answer written in that same format. This is the one
      that costs the campaign real reward: if the verifier cannot parse the
      format, a correct answer earns 0 in GRPO forever. The fix must make these
      CORRECT, not merely judgeable.

Run:  .venv/bin/python scripts/campaign-20260925/probe_verifier_gaps.py
"""

from adaptive_math.core.types import AnswerType, ReferenceAnswer
from adaptive_math.verifier.service import verify_answer

CASES: list[tuple[str, str, str, str, str]] = [
    # (class, polarity, prediction, reference, note)
    (
        "open_series_ellipsis",
        "negative",
        r"\frac{1}{\sqrt{5}} + \frac{2}{5} + \frac{3}{25} + \cdots",
        r"(2i-1)/4",
        "omni_math:02f104ee",
    ),
    (
        "yes_no_vs_structure",
        "negative",
        r"\mathbb{Z}_{2^{1005}}\times \mathbb{Z}_{2^{1005}}\times \cdots",
        "No",
        "omni_math:0986e00a group decomposition vs No",
    ),
    ("yes_no_vs_structure", "positive", "No", r"\text { No }", "correct yes/no answer"),
    ("yes_no_vs_structure", "positive", "Yes", r"\text{Yes}", "correct yes/no answer"),
    (
        "relation_vs_value",
        "negative",
        r"b^2 - 4ac \geq 0",
        r"49 \pi",
        "omni_math:0d9e9393 inequality vs area",
    ),
    ("relation_vs_value", "positive", r"x \geq 2", r"x \geq 2", "same relation"),
    ("relation_vs_value", "positive", r"2 \leq x", r"x \geq 2", "relation written reversed"),
    (
        "statement_vs_no",
        "negative",
        "a=b",
        r"\text { No }",
        "omni_math:0adb65d7 equation vs No",
    ),
    (
        "multi_value_list",
        "negative",
        r"90\degree ,60\degree ,30\degree",
        r"60^\circ",
        "omni_math:0a28be89 triple vs single angle",
    ),
    ("multi_value_list", "positive", r"60\degree", r"60^\circ", "\\degree must equal ^\\circ"),
    (
        "huge_binomial",
        "negative",
        r"\binom{4032}{2016}",
        r"\binom{4030}{2015}",
        "omni_math:0a6a00a1 unevaluated huge binomials",
    ),
    (
        "huge_binomial",
        "positive",
        r"\binom{4030}{2015}",
        r"\binom{4030}{2015}",
        "identical huge binomial must be correct",
    ),
    (
        "factorial_power",
        "negative",
        r"8!/(2!^4)",
        "105",
        "omni_math:0c22491e = 2520, wrong",
    ),
    (
        "factorial_power",
        "positive",
        r"8!/(2!^4 \cdot 4!)",
        "105",
        "the SAME syntax used correctly must earn reward",
    ),
]


def main() -> None:
    for gap_class, polarity, prediction, reference, note in CASES:
        verdict = verify_answer(
            prediction,
            ReferenceAnswer(value=reference, answer_type=AnswerType.EXPRESSION),
            task_id=f"probe:{gap_class}:{polarity}",
        )
        reason = verdict.details.get("reason") or verdict.details.get("error_code") or ""
        flag = "<<" if polarity == "positive" and verdict.status.value != "correct" else ""
        print(
            f"{gap_class:22s} {polarity:8s} -> {verdict.status.value:18s} "
            f"reason={str(reason)[:52]:52s} {flag}"
        )
        if verdict.normalized_prediction is not None:
            print(f"{'':22s} {'':8s}    pred-> {verdict.normalized_prediction[:70]}")


if __name__ == "__main__":
    main()
