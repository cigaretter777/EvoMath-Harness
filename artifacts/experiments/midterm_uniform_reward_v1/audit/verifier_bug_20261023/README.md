# Verifier set-membership crash — diagnosis and fix

- Discovery time: `2026-10-22T19:48:20Z`
- Discovery context: first formal unified evaluation, group `sft`, task
  `math:45e5f7656e796fc9615c`, after 59 valid SFT trajectories.
- Original verdict: `internal_error`, reason `comparator raised TypeError`.
  The evaluator retried the verifier once; the same error recurred, so the
  evaluator retained the invalid row and paused with exit code 3.

## Trigger

The model answered:

```text
x \in (0, 36)
```

The reference is:

```text
(0,9) \cup (9,36)
```

`latex2sympy` parses the prediction as `Eq(x, Interval.open(0, 36))` and the
reference as `Union(Interval.open(0, 9), Interval.open(9, 36))`.  Both are
`Relational`/`Set` objects rather than arithmetic expressions.
`_relation_signature` unconditionally computed `lhs - rhs`, which raised:

```text
TypeError: unsupported operand type(s) for -: 'Symbol' and 'Interval'
```

The exception escaped `compare_expressions`, whose contract says it never
raises, and the worker reported `internal_error`.

The same failure mode occurs for tuple equalities such as
`(p,q) = (1,1)` (`Tuple - Tuple` TypeError).  Those appeared in the training
rollout audits as well.

## Impact found in completed training (diagnostic only)

Training rollout audits contain `internal_error` statuses:

| group | internal_error steps | count |
|---|---:|---:|
| r0 | 2 | 2 |
| r1 | 4 | 4 |
| r2 | 3 | 3 |

Among the nine, two are semantically correct under the fixed comparator:

- r1 `math:163a181d4fc954c57e62`: `k = -3, 0` against reference `0,-3`
- r1 `math:530dbff74491f8eca29f`: `x \in (-∞,-2) ∪ (7,∞) ∪ (-1,1) ∪ (2,3) ∪ (4,6)`
  against reference `(-∞,-2) ∪ (-1,1) ∪ (2,3) ∪ (4,6) ∪ (7,∞)`

The other seven were wrong answers.  Training is not rerun retroactively: the
three 50-step groups completed with the original verifier and the training
budget is already committed.  The evaluation phase uses the fixed comparator
for every group; this training-side false-negative rate is disclosed as a
limitation in the experiment report.

## Fix

Minimal source change in `src/adaptive_math/verifier/symbolic.py`:

1. `_relation_signature` catches non-arithmetic operand errors
   (`TypeError`, `ValueError`, `AttributeError`, `ArithmeticError`,
   `NotImplementedError`) and returns `None`, allowing structural comparison.
2. Set answers (`Interval`, `Union`, `FiniteSet`, ...) are compared with a
   symmetric difference; undecidable emptiness returns an unverifiable
   prediction rather than a guess.
3. Set-membership answers (`Eq(symbol, Set)` or `Contains(symbol, Set)`) are
   compared by their target set.  The variable side must be a single symbol,
   so `x^2 \in S` is not incorrectly treated as the set `S`.
4. Tuple equalities no longer crash; they fall through to the existing
   structural path and return a verdict.

No training code, data, reward configuration, model, or evaluation protocol
was changed.

## Hashes and evidence

- Original module:
  `f4f21ac39e66ee21ae8fea4625b9a9522a7e0f92747b6ac3bc1fef393a6b2194`
- Fixed module:
  `1de775613d9cb3fe8a9e0c91eacb59b239ac3609ee7c15265f2b579f83088fd7`
- Backup: `symbolic.py.pre_fix` in this directory.
- Unified diff: `symbolic_set_comparison.patch`
  (`5a926f0f5b143fe8444405d9ad98ab072158d6e880bebe0b865e4315581c13b0`)
- Direct regression tests in `tests/unit/verifier/test_format_coverage_gaps.py`:
  `test_set_membership_answers_are_compared_as_sets`,
  `test_non_arithmetic_relational_operands_do_not_raise`.
- Verifier test run: `40 passed` for `test_format_coverage_gaps.py`,
  `test_symbolic.py`, and `contract/test_verifier_worker.py`.
- Isolated-worker reproduction now returns:
  `incorrect` with `details.method=membership` for the paused task.

## Evaluation action

The 59 SFT rows collected before the fix are retained in
`evaluation/formal_unified_tool_eval_20261023/` as evidence but are not used.
A fresh formal run is started in a new directory so every group is evaluated
with the same fixed verifier.  The old run is never resumed or mixed with the
new one.
