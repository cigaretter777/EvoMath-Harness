# Trajectory spot-check: success / wrong / invalid action / tool call

**Date:** 2026-09-29
**Scope:** criterion 6 of the MVP design's §6.3 —
"至少抽查成功、错误、非法动作和工具调用四类轨迹"
(at least spot-check success, wrong-answer, invalid-action and tool-call
trajectories). One exemplar per category from the stored MVP arms, plus a
mechanical census of **every** invalid action in all three arms.
**CPU-only:** no GPU, no generation, no re-scoring. Everything below is re-read
from artifacts on disk; §5 is a re-parse, not an interpretation.

**Data** — the same three runs the arm table and the packaging doc report:

| arm | file | sha256 |
|---|---|---|
| base | `artifacts/rollout_health/thesis_e0_base_tool/trajectories.jsonl` | `5827e739dbdd1054bdda861a703a1d605212ecf7442af0034670d7871ce9071a` |
| sft | `artifacts/rollout_health/thesis_e0_sft_tool/trajectories.jsonl` | `1cae7cf8b0c42f5281116b782209963ba0b741825c37978e0aee69f27beb09ae` |
| grpo | `artifacts/rollout_health/thesis_e0_r0_tool/trajectories.jsonl` | `cb727b5ba4813c131e5d211c6a6ee4631d430db0c4ac6d99d08bb3630843341b` |

**Exemplar selection** (deterministic rules over the 200 tasks per arm; full
predicates so anyone can re-run them):

| category | rule | task |
|---|---|---|
| success | first up-pair in lexicographic task-id order: `sft` correct and `base` not, without an executed base tool call (19 of 20 qualify) | `omni_math:009736eb48db11e4ba52` |
| wrong answer | lexicographically first `grpo`-incorrect task | `omni_math:00074dca01ab2ba8c9e2` |
| invalid action | lexicographically first `base` task with the arm-maximum 6/6 invalid steps and no tool calls | `omni_math:1052a4167c88e936e057` |
| tool call | lexicographically first `base` task with the arm-maximum 3 executed tool calls | `omni_math:092a091fc158681e291f` |

The two rejection rules quoted below live in `src/adaptive_math/agent/parser.py`
(`parse_action`) and `src/adaptive_math/agent/actions.py` (`FinalAction`), and
the runs used **strict mode**: `AgentLoop`'s default
`parser_tolerance=()` (`src/adaptive_math/agent/loop.py`), so the opt-in
tolerance rules (`unclosed_think`, `bare_final_scalar`) were **not** applied.
`parser.py`, `actions.py` and `loop.py` are unchanged since `74c61da` (the
commit the arms ran), so re-parsing at HEAD decides exactly what the runner
decided then — and §5 confirms it, row by row.

## 1. Success — `sft`, `omni_math:009736eb48db11e4ba52`

`verdict = {"status": "correct", "reward": 1.0, "details": {"method": "canonical"}}`,
usage `steps=1, invalid_actions=0, tool_calls=0`. One protocol turn, closed:

```text
… The sum of all such two-digit positive integers $x$ is: 15 + 21 + 35 = 71
</think>

<final>{"answer":"71"}</final>
```

**Reading.** The cheap half of the SFT story, in one row: a clean one-turn
close, no invalid actions, verifier-correct. The same task shows what `base`
failed to do with it — 6/6 invalid steps, `max_steps`, no final answer, even
though one of its turns contained a well-formed-looking
`<final>{"answer":71}</final>` (rejected as `invalid_schema`: `FinalAction.answer`
is typed `str`, and `71` is a JSON number) and another derived 71 in prose.
Base had the task in reach and could never submit it; the SFT arm's difference
on this row is exactly "close legally", not "compute better".

## 2. Wrong answer — `grpo`, `omni_math:00074dca01ab2ba8c9e2`

`verdict = {"status": "incorrect", "normalized_prediction": "y/9", "details":
{"method": "structure", "reason": "free symbol mismatch"}}`,
usage `steps=1, invalid_actions=0, tool_calls=0`:

```text
… The final answer is \(\boxed{\frac{Y}{9}}\).</think>

<final>{"answer":"\\frac{Y}{9}"}</final>
```

**Reading.** The protocol half of the loop runs perfectly — one turn, legal
envelope, answer extracted — and the answer is still wrong: the prediction
carries a free symbol, the reference carries none, and the structural verifier
says so. This is the concrete face of the report's H2 distinction: format
discipline and correctness are two different axes, and the trained arms moved
far on the first without moving the second in step. (`base` on the same task
reproduces its own failure shape: 6/6 invalid, including
`}<final>{"answer": "Y/(3*sqrt(3))"}>` — one stray brace after `</final>` makes
the envelope not match, `invalid_envelope`.)

## 3. Invalid action — `base`, `omni_math:1052a4167c88e936e057`

`verdict = invalid_prediction ("no_final_answer")`, termination `max_steps`,
usage `steps=6, invalid_actions=6, tool_calls=0, generated_tokens=6144` — every
step invalid, every step hitting the 1024-token cap. Turn 8 of 6, verbatim
start:

```text
<think>
</think>

<tool_call>{"name":"sympy","arguments":{"operation":"solve","expression":"r(n, 1000) > r(n, 1001)"}}

<tool_call>{"name":"sympy","arguments":{"operation":"solve","expression":"r(n, 1000) > r(n, 1001)"}}

… (28 identical unclosed blocks in this one turn; none ever closed with
</tool_call>, and the turn is truncated mid-envelope by the token cap)
```

**Reading.** "Invalid action" is not only "no action at all". Here the model
keeps *starting* the envelope and never finishes it; the parser counts action
tags before matching, so 28 opens in one turn is `multiple_actions` outright
(the same turn-2 shape with two opens also lands there). The loop's message —
"Use exactly one `<tool_call>` or `<final>` action." — is delivered five times
and changes nothing. Base's 1118 invalid actions are a mixture of exactly these
shapes; §5 counts them.

## 4. Executed tool call — `base`, `omni_math:092a091fc158681e291f`

`verdict = invalid_prediction ("no_final_answer")`, `max_steps`, usage
`steps=6, tool_calls=3, invalid_actions=3`. This one task holds 3 of the base
arm's 7 executed tool calls (7 across 200 tasks). The executed exchange, verbatim:

```text
<tool_call>{"name":"sympy","arguments":{"operation":"solve","expression":"1/6 + 1/3 - 1/x == 0","variables":["x"]}}</tool_call>
tool_result {"ok": true, "output": "[]", "error_code": null, "latency_ms": 0, …}
```

The identical call is issued three times over three steps; after the first
result the model's own thinking reads:

```text
… adding 1/6 and 2/6 gives 3/6, which simplifies to 1/2. So the equation
becomes 1/2 = 1/x. To find x, I can cross-multiply …
```

**Reading.** The tool executes cleanly (ok, ~0 ms), returns an empty solution
list, and the loop still dies at `max_steps` with no final answer — the answer
was derived in prose and never submitted, and the tool call contributed nothing
to the outcome. This is the strongest concrete form of the report's §9 caution
("cannot attribute the accuracy gain to tool execution"): the arm that executes
tools gets 0 for this task, while `sft` and `grpo` both solve it in one turn
with zero tool calls. (Whether `[]` is the correct reading of that equation by
the demo service is not adjudicated here; the exemplar records the loop, not
the service.)

## 5. Census: every invalid action, attributed

The trajectory stores only the generic message for an invalid action, not the
parse error code. So each of the three files was replayed through the tree's
own `parse_action` (strict), and each `model_output`'s recorded successor was
compared against the re-parse:

**Zero disagreements, 1888/1888 turns.** Across the three arms, 1888 model
outputs decompose exactly into 1507 invalid actions + 381 accepted actions
(34 base = 27 finals + 7 tool calls; 159 sft finals; 188 grpo finals — the
final-action counts of the arm table). Every accepted action re-parses to the
action the run recorded; every rejected one re-parses to `None`. The per-arm
invalid totals equal `sum(usage.invalid_actions)` exactly.

| arm | invalid | `invalid_envelope` | `multiple_actions` | `invalid_schema` | `invalid_json` |
|---|---:|---:|---:|---:|---:|
| base | 1118 | 960 | 74 | 84 | 0 |
| sft | 274 | 165 | 2 | 106 | 1 |
| grpo | 115 | 97 | 6 | 6 | 6 |

The `invalid_schema` rows decompose further:

* **base, 79 of 84** — `<final>` with a non-string `answer`
  (`{"answer":71}`-style: the envelope is perfect, the JSON type is not);
* **base, 5 of 84** — a final-style payload inside a tool envelope
  (`<tool_call>{"answer":"12"}</tool_call>`: no `name`, so `ToolCall` fails);
* **sft 106, grpo 6** — bare-scalar finals (`<final>74</final>`), the exact
  shape the campaign's opt-in `bare_final_scalar` tolerance exists for, and
  which the strict runs do not rescue.

So the trained arms' remaining invalid actions are dominated by *retried but
unrescuable* shapes they eventually abandon (sft still reaches a legal final on
159/200 tasks), while base's are dominated by envelopes that never close.

## 6. What this does not claim

* Four exemplars are not a generalization, and none of them is a random draw —
  the rules above are deterministic and re-runnable, and §5's census is
  exhaustive rather than sampled.
* No causal claim about SFT/GRPO, the tools, or the prompt; the exemplars are
  read as the loop recorded them.
* No re-scoring, no lenient re-extraction; the verdicts quoted are the stored
  strict ones, and the lenient column of the report is a separate disclosure.
* The success/wrong exemplars were chosen to be *illustrative of the recorded
  verdicts*, not to be representative; the counts in the arm table remain the
  quantitative claim.

## 7. Reproduce

```python
# one-off, from the repo root with the venv (needs adaptive_math importable)
import json
from adaptive_math.agent.parser import parse_action   # strict: no tolerance

def show(arm_file, task_id):
    for line in open(arm_file):
        r = json.loads(line)
        if r["task_id"] == task_id:
            for e in r["trajectory"]["events"]:
                p = json.dumps(e["payload"])
                print(e["sequence"], e["kind"], p[:200])
            print(r["verdict"], r["trajectory"]["usage"])
```

* the four exemplars: call `show` with the file/arm and task ids in the table
  at the top;
* the §5 census: for every `model_output`, take its successor event's kind and
  compare `parse_action(raw).action is None` against it — the two must agree
  1888/1888 (any disagreement is a finding, not a footnote);
* file hashes: `sha256sum artifacts/rollout_health/thesis_e0_{base,sft,r0}_tool/trajectories.jsonl`.
