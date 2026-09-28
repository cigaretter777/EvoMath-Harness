# Protocol gap in the stored 2026-09-26 agent arms

**Date:** 2026-09-29
**Arms re-parsed:** `artifacts/rollout_health/thesis_e0_base_tool` (base + tools),
`artifacts/rollout_health/thesis_e0_rule_strategy` (fixed-rule strategy)
**Script:** `scripts/analysis/protocol_gap.py` (tests: `tests/unit/analysis/test_protocol_gap.py`)
**Output:** `artifacts/results/mvp-20260929/protocol-gap.json`,
sha256 `90c5d2dfd0fa7306f5969539f6875c00fe43fd97dd3c9392d2018ec28758262c`
(the command below reproduces this file byte for byte; a hash quoted in an
earlier revision of this doc came from before the `coerce_types` rung was added)
**No GPU, no regeneration.** Every number below is a re-parse of raw model turns a
stored arm already produced.

```bash
python scripts/analysis/protocol_gap.py \
  --arm base_tool=artifacts/rollout_health/thesis_e0_base_tool/trajectories.jsonl \
  --arm rule=artifacts/rollout_health/thesis_e0_rule_strategy/trajectories.jsonl \
  --out artifacts/results/mvp-20260929/protocol-gap.json
```

## 1. The funnel is the runtime's own accounting, not a reinterpretation

The strict protocol is a `fullmatch` of `<think>` (optional) plus exactly one
action block whose payload satisfies its tag's schema. Re-parsing gives, for
every arm:

| arm | turns | executed tool | executed final | turns − executed | recorded `invalid_actions` |
|---|---:|---:|---:|---:|---:|
| base_tool | 1152 | 7 | 27 | 1118 | 1118 |
| rule | 721 | 16 | 16 | 689 | 689 |

The identity in the last two columns is exact, and it is the reason to trust the
breakdown that follows: **every turn that did not produce an executable action
was recorded as an invalid action**, so the classification below is not a second
opinion about the run -- it is the run's own invalid-action count, attributed by
cause. (Self-check: the script raises if its re-parse of executed tool calls
disagrees with any trajectory's recorded `usage.tool_calls`.)

## 2. The envelope, not tool aversion

The base+tool arm records 7 executed tool calls over 200 tasks. Read alone, that
says the model does not call tools, and it was read that way -- it is the reason
the `sft_dp_v2` plan tried to inject sympy demonstrations into the SFT data.
Re-parsing says otherwise:

| arm | turns | tool attempts | executed tool calls | exec. rate | tasks with a tool attempt | tasks one repair from execution |
|---|---:|---:|---:|---:|---:|---:|
| base_tool | 1152 | 422 | 7 | 0.6% | 200/200 | 193/200 |
| rule | 721 | 249 | 16 | 2.2% | 124/124 | 109/124 |

The model attempted a tool action in every one of the 200 tasks, and in 193 of
them the attempt had a schema-valid payload. The execution rate is a property of
the envelope, not of the intent. The largest single cause is the smallest
possible defect: a complete, schema-valid payload whose closing tag was never
emitted.

## 3. Failure taxonomy

Classes partition the turns (a turn's class is decided by its first action tag).
Counts for both arms:

| class | base_tool | rule | what it is |
|---|---:|---:|---|
| `no_action` | 609 | 385 | reasoning with no action tag at all |
| `tool_unclosed_payload_valid` | 309 | 156 | complete valid payload, closing tag missing |
| `final_closed_payload_invalid` | 80 | 52 | well-formed closed block, payload fails the schema |
| `tool_unclosed_trailing_content` | 70 | 32 | payload valid, a second action stacked in the same turn |
| `executed` | 34 | 32 | strict parse produced the action (7+27 / 16+16) |
| `tool_unclosed_truncated` | 12 | 39 | body starts as JSON and is cut off mid-value |
| `tool_unclosed_payload_invalid` | 11 | 3 | payload decodes, fails the schema |
| `tool_closed_payload_valid` | 8 | 3 | closed and valid, but not alone in the turn |
| `tool_closed_payload_invalid` | 5 | 0 | closed block, schema-invalid payload |
| `final_closed_payload_not_json` | 5 | 6 | closed block whose body is not JSON |
| `final_unclosed_payload_valid` | 3 | 0 | final payload valid, closing tag missing |
| `final_unclosed_trailing_content` | 2 | 6 | final payload valid with trailing text |
| `final_unclosed_truncated` | 2 | 6 | final body cut off mid-value |
| `final_mentioned_in_prose` | 2 | 0 | tag discussed, no payload |
| `tool_mentioned_in_prose` | 0 | 1 | tag discussed, no payload |

Two readings of the `final_closed_payload_invalid` row matter. Measured on the
base+tool arm, 77 of those 80 are a JSON *number* where `FinalAction.answer`
requires a string (`{"answer":71}`); the other 3 are floats. The model closed a
well-formed final block carrying the right value and the run discarded it on a
type. That is a different repair from closing a tag -- and it is not specific to
the agent arms, see §6.

## 4. Three repair rungs (escalating ceilings)

Each rung contains the one below it, so a turn counted at `auto_close` is also
counted at the rungs above. `executed` (the strict reading) is contained in all
of them.

| rung | definition | base_tool (final / tool) | rule (final / tool) |
|---|---|---:|---:|
| `executed` | the strict parse the run used | 27 / 7 | 16 / 16 |
| `auto_close` | close the missing tag, change nothing else | 7 / 309 | 0 / 156 |
| `extract_first_block` | + take the first block, ignore prose and stacked actions | 42 / 393 | 31 / 206 |
| `coerce_types` | + read a scalar answer literal as its string form | 123 / 393 | 90 / 206 |

Read as ceilings: on the base+tool arm, 123 of 1152 turns carry a final action
that a mechanical, information-preserving repair would have executed, against
27 the strict run executed. Tool-side, 393 turns against 7.

## 5. Verbatim examples

Dominant shape, `tool_unclosed_payload_valid` (task `omni_math:00074dca01ab2ba8c9e2`,
149 characters, whole turn, `\n` written out):

```
<think>\n</think>\n\n<tool_call>{"name":"sympy","arguments":{"operation":"solve","expression":"3*sqrt(Y/(3*sqrt(3))) == sqrt(Z)","variables":["Y","Z"]}}
```

Stacked actions, `tool_unclosed_trailing_content` (task `omni_math:004a1425276f6b04af18`,
first 210 of 2854 characters):

```
<think>\n</think>\n\n<tool_call>{"name":"sympy","arguments":{"operation":"solve","expression":"E = 1 + (1/2) * E1 + (1/2) * E2"}}\n\n</think>\n\n<tool_call>{"name":"sympy","arguments":{"operation":"solve","expression":"E = 1 + …
```

Scalar answer, `final_closed_payload_invalid` (task `omni_math:009736eb48db11e4ba52`,
tail of 1779 characters):

```
answer is 71. Therefore, the answer should be 71.\n</think>\n\n<final>{"answer":71}</final>
```

Closed-and-valid but not alone, `tool_closed_payload_valid`
(task `omni_math:041202f5053c3139ed67`, tail of 1539 characters):

```
 left, divide by 6. So x is 4. I think that's it.\n</think>\n\n<tool_call>{"name":"python","arguments":{"code":"result = (2 * 3 * 4) / 6; result"}}</tool_call>\n<tool_call>{"answer":"4"}
```

The last one is worth reading twice: the first block is a legal tool call, and
the turn ends with the answer in a second `<tool_call>` block rather than
`<final>` — the full-match envelope rejects a turn that a human would read as a
correct trajectory.

## 6. Correction: not the direct arms

This section predicted that the scalar-answer shape would also be found in the
direct arms' stored generations. Re-scoring them under the same ladder refutes
that: coercion moves every direct arm by exactly **0 rows**, and the reason is
mechanical rather than statistical.

* `base_direct` emits no `<final>` tag in any of its 200 generations (15 use
  `<tool_call>` instead, 7 contain a numeric answer inside *some* wrapper, none
  inside a well-formed final block), so there is nothing for the rung to read.
* `sft_direct`, `r0` and `r2` emit `<final>` in 133/148/132 generations, and not
  one of them writes the answer as a number.

The shape is therefore an artifact of the agent prompt's JSON contract, not a
general property of the models. The consequence is the useful part: the strict
column for the direct arms is a **mathematics** number (SFT satisfies the wire
format 133 times and converts it into 27 correct answers), while for the agent
arms the strict column is substantially a **schema-conformance** number (22 of
`base_agent`'s 35 readable last-turn answers are refused for typing `answer` as
a number). Full table and paired tests: §6 of
`docs/results/mvp-20260929-arm-table.md`; artifact
`artifacts/results/mvp-20260929/tolerance-rescore.json`.

## 7. What this does **not** claim

* **Repairs are counterfactuals over recorded text, not scores.** Executing a
  repaired tool call produces an observation the model never saw, so the
  trajectory diverges immediately after the repair. The rungs bound what the
  envelope cost; they do not predict what the arm would have scored.
* **Execution is not correctness.** Even at `coerce_types`, the final answer
  still has to verify. The funnel bounds how many actions the protocol discarded
  -- it says nothing about whether those answers were right.
* **`no_action` is not a protocol artifact.** 609 of 1152 base+tool turns and
  385 of 721 rule turns carry no action tag at all. That is the model choosing
  to keep reasoning (and, on the direct arm, being truncated at 1024 tokens),
  which no envelope repair reaches.
* **The rungs are ceilings, not a ranked plan.** `auto_close` is the cheapest
  and least interpretive repair; `extract_first_block` and `coerce_types` change
  more about the run and should be judged separately.

## 8. What it changes

The next step is no longer "teach the model to use tools". The stored arms show
a model that reaches for tools constantly and a protocol that discards the
reach. Two consequences:

1. The planned `--parser-tolerance unclosed_think,bare_final_scalar` fork in
   `scripts/eval/run_agent_eval.py` is not a convenience flag; it is the
   difference between measuring tool use and measuring tag emission. The rungs
   above quantify what each tolerance buys **before** any GPU hour is spent.
2. Any claim that the agent arm "barely used tools" must be restated as "the
   strict envelope executed 7 of 422 attempts" -- the first phrasing is a
   property of the model, the second is a property of the measurement.
