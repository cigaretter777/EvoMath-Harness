# Invalid-prediction Audit (campaign step 1)

Scope: all 195 `invalid_prediction` rows across the three eval arms
(SFT 71, R0 55, R2 69) on frozen OmniMath-200. Classifier:
`scripts/campaign-20260919/classify_invalid.py`; raw samples stay on the
AutoDL host under `artifacts/audit/invalid-classification-20260919/`.

## Category distribution

| Category | sft | r0 | r2 | Total | Share of invalid |
|---|---:|---:|---:|---:|---:|
| truncated_at_cap (output_tokens = 1024) | 67 | 51 | 68 | 186 | 95.4% |
| verifier_rejected_parseable (extract ok) | 4 | 4 | 1 | 9 | 4.6% |
| final_malformed / ambiguous_tail / no_submission / empty_or_garbage | 0 | 0 | 0 | 0 | 0% |

Notes:

- `finish_reason` is a stub (`transformers_client.py:78` hardcodes
  `"stop"`), so truncation is detected from `output_tokens == 1024`
  (the eval `max_new_tokens`), not from the field.
- Cross-arm: 30 tasks are invalid in all three arms (stable
  long-reasoning/hard tasks), 36 in two, 33 in one. The model protocol
  itself is fine: whenever it reaches `<final>`, extraction succeeds.

## Sub-findings

1. **27.4% of truncated tails contain repetition loops** (a ≥24-char
   phrase repeated ≥3×). The truncated bucket splits into genuine
   long-reasoning budget exhaustion (~73%) and decode degeneration
   burning the budget in circles (~27%).
2. **Verifier coverage gaps.** All 9 `verifier_rejected_parseable` rows
   were checked against their references: 0 of 9 were actually correct
   (e.g. the model answered `a=b` where the reference is "No", and in
   one case its own reasoning said "not necessarily true"). Six gap
   types identified: symbolic series with `…`, group-structure answers
   vs a "No" reference, multi-value lists, huge binomials, symbolic
   statements vs "No", inequality/condition answers (plus `n!^k`
   factorial syntax). Current impact is label quality only (wrong
   answers mislabeled invalid instead of incorrect); the risk is future
   rounds where a correct answer in one of these formats would earn
   zero reward in GRPO.

## Triage

| Audit finding | Route |
|---|---|
| Dominant: truncation / no submission | Harness patch direction (Evo round-1 hypothesis: reduce `TRUNCATED_AT_CAP`) |
| Verifier coverage gaps (small, no false negatives) | Fix the judging/reward pipeline before the RL pilot |
| Valid-format-but-wrong answers | None found — no hard-example pool input from this audit |
| Mixing of failure causes | Low: two categories cover 100% of invalids |

Candidate single-field harness patches for Evo round 1 (pick ONE):

- `max_new_tokens` 1024 → 2048 for eval generation;
- prompt instruction to submit `<final>` earlier (budget-awareness);
- decode-side repetition penalty for the loop sub-population.
