# R2 continuation: SFT / R2-50 / R2-75 / R2-100

Primary target: correctness and tool cost. Average reward and protocol
diagnostics are shown only as secondary diagnostics.

## Main table

| group | correct | calls/task | tool s/task | tool failure | invalid act. | truncation | 0 calls | 1 call | 2+ calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | 110/200 (55.00%) | 0.9900 | 0.4701 | 19.70% | 1.00% | 1.00% | 1.00% | 99.00% | 0.00% |
| R2-50 | 108/200 (54.00%) | 0.9850 | 0.6070 | 20.30% | 1.98% | 1.00% | 1.50% | 98.50% | 0.00% |
| R2-75 | 112/200 (56.00%) | 0.9850 | 0.4545 | 17.26% | 0.75% | 1.00% | 1.50% | 98.50% | 0.00% |
| R2-100 | 108/200 (54.00%) | 0.9950 | 0.4551 | 21.61% | 6.15% | 0.00% | 0.50% | 99.50% | 0.00% |

## Paired comparisons (paired by task, 200 pairs)

| comparison | Δcorrect (95% CI) | McNemar p (Holm) | Δcalls/task (95% CI) | Δtool s/task (95% CI) |
|---|---:|---:|---:|---:|
| R2-75 - R2-50 | +0.0200 [-0.0050, +0.0500] | 0.289062 (0.867188) | +0.0000 [-0.0200, +0.0200] | -0.1525 [-0.4556, +0.0019] |
| R2-100 - R2-50 | +0.0000 [-0.0300, +0.0300] | 1 (1) | +0.0100 [+0.0000, +0.0250] | -0.1519 [-0.4548, +0.0025] |
| R2-100 - R2-75 | -0.0200 [-0.0500, +0.0100] | 0.34375 (0.867188) | +0.0100 [+0.0000, +0.0250] | +0.0006 [-0.0058, +0.0073] |
| R2-50 - SFT | -0.0100 [-0.0400, +0.0200] | 0.753906 (1) | -0.0050 [-0.0250, +0.0100] | +0.1369 [-0.0236, +0.4435] |
| R2-75 - SFT | +0.0100 [-0.0200, +0.0400] | 0.753906 (1) | -0.0050 [-0.0250, +0.0100] | -0.0156 [-0.0360, -0.0017] |
| R2-100 - SFT | -0.0100 [-0.0450, +0.0250] | 0.774414 (1) | +0.0050 [+0.0000, +0.0150] | -0.0150 [-0.0349, -0.0018] |

## Protocol check

PASS: the continuation run matches the v2 formal run on the frozen protocol keys.

The evaluator script hash differs only because the continuation group
selection was added; generation, parser, budget, verifier and prompt
configuration are checked to be identical to the v2 run.

## Caveats

- Steps 1-50 used the pre-fix verifier; steps 51-100 used the fixed
  verifier. This is a performance exploration, not a pure step-count
  ablation.
- The resume reuses the same 400-row schedule in a second epoch; steps
  51-100 are not fresh data relative to steps 1-50.
- The vLLM sampling RNG and tool-time jitter restart at resume; do not
  interpret small differences as a causal step-count effect.
- Reward means and protocol metrics are diagnostics, not the target.

