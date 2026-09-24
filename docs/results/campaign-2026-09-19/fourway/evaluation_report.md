# Four-way Paired Evaluation (offline join)

- Tasks: 200 (OmniMath `frozen_eval`, task_ids_sha256 `1fc257f2…`)
- Prompt version: `agent-v1`
- Verifier/extractor/prompt/parquet hashes identical across all arms (checked against eval manifests), so per-task verdicts are directly comparable.
  - **Correction, 2026-09-24:** the hash check above is correct but incomplete.
    The SFT baseline arm was generated at `batch_size=8` and the R0/R2 arms at
    `batch_size=1`. Reproducing the baseline's model identity at batch 1 shows
    greedy decoding is not batch invariant on this stack: raw outputs are
    byte-identical on only 3/200 tasks, verifier labels agree on 130/200, and
    correct/incorrect agrees on 185/200 — i.e. batching alone moves 15 tasks,
    which is larger than every delta reported in the table below (−5, −1/+1, +3
    improved-vs-regressed). The null conclusions stand; any directional reading
    of a few-task difference between the batch-8 baseline and the batch-1 RL
    arms does not. See `docs/results/campaign-2026-09-24/report.md` §3.

| Arm | Correct | Invalid prediction | Valid-answer rate |
| --- | ---: | ---: | ---: |
| SFT baseline (09-14, dp_v1 merged) | 27/200 | 71 | 64.5% |
| R0 adapter (50-step GRPO, answer-only reward) | 22/200 | 55 | 72.5% |
| R2 adapter (12-step GRPO, tool-weighted reward) | 25/200 | 69 | 65.5% |

| Pair | Δ accuracy | Improved | Regressed | Unchanged | McNemar p | Bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT → R0 | -2.5% | 5 | 10 | 185 | 0.302 | [-6.0%, +1.0%] |
| SFT → R2 | -1.0% | 1 | 3 | 196 | 0.625 | [-3.0%, +1.0%] |
| R0 → R2 | +1.5% | 8 | 5 | 187 | 0.581 | [-2.0%, +5.0%] |

## Conclusion

None of the three pairwise differences is significant (McNemar exact, paired
bootstrap). Both RL adapters are statistically indistinguishable from the SFT
baseline on frozen OmniMath-200. The only visible signal is R0's protocol
conformance improvement: invalid predictions dropped from 71 to 55
(valid-answer rate 64.5% → 72.5%) without an accuracy change.

The R0/R2 numbers are consistent with the 09-14 SFT baseline under identical
verifier/extractor/prompt hashes, so the low absolute accuracy (~11–13.5%) is
a property of the 1.7B model + strict protocol on this pool, not an artifact of
a broken eval pipeline.
