# Source drift justification — `base+tool` arm (`fdf9276`) → this branch

**Purpose.** The three-arm agent evaluation pairs new adapter arms against the
stored `base+tool` arm at `artifacts/rollout_health/thesis_e0_base_tool/`, which
was recorded at commit `fdf9276` (2026-09-26). Between that commit and the HEAD
this evaluation runs from, **eight files under the evaluation path changed**.
`scripts/eval/verify_arm_identity.py` refuses to pass while that drift is
unjustified, and this document is the justification it demands.

It is written to be audited, not to be trusted. Every claim below is stated so
that a reader can re-derive it with the command given beside it.

## 1. What changed, and by which commit

```
$ git log --oneline fdf9276..HEAD -- src/adaptive_math/agent src/adaptive_math/tools \
      src/adaptive_math/verifier src/adaptive_math/evaluation src/adaptive_math/tasks configs
b273d29 feat: harness evolution foundations - H1 spec binding + H2.5 failure taxonomy
0feb442 merge: origin/main (three campaigns + verifier/parser/reward fixes) into V3 foundations
d992c9f Fix CI: lint/type errors surfaced by the first main-branch CI run
```

Three commits: the V3 harness foundations, the merge that brought the 09-24/09-26
verifier and parser fixes onto the trunk, and a lint/type cleanup. None of them
is a change to the agent's decision-making; the question is whether any of them
changed it *incidentally*.

## 2. Audit, file by file

```
$ git diff --stat fdf9276 HEAD -- src/adaptive_math/agent src/adaptive_math/tools \
      src/adaptive_math/verifier src/adaptive_math/evaluation src/adaptive_math/tasks configs
 configs/harness/champion_v1.yaml       | 25 +++++++++++++++++++++++++
 src/adaptive_math/agent/environment.py | 28 ++++++++++++++++++++++------
 src/adaptive_math/agent/loop.py        | 29 +++++++++++++++++++++++++++--
 src/adaptive_math/agent/parser.py      |  3 ++-
 src/adaptive_math/agent/replay.py      | 16 ++++++++++++++--
 src/adaptive_math/agent/state.py       | 26 ++++++++++++++++++++++++--
 src/adaptive_math/agent/trace.py       | 10 +++++++++-
 src/adaptive_math/tools/registry.py    |  5 +++++
```

**`agent/parser.py`** — a `cast(list[JSONValue], applied)` on the `details` dict.
A type annotation; `cast` is a runtime no-op. No parsing behaviour changes.

**`agent/state.py`** — adds a `StepErrorCode` enum and an
`error_code: StepErrorCode | None = None` field on `TraceEvent`, plus a
keyword-only `error_code` parameter on `append_event`. The field's own comment
states the invariant that matters here: *"Excluded from the canonical trajectory
hash when None so pre-existing content hashes never change."*

**`agent/trace.py`** — adds `harness_spec_hash` and `model_version`, both
defaulting to `None`, both carrying the same exclusion rule in their comments.

**`agent/replay.py`** — implements that rule: `trajectory_hash` deletes each of
the three fields from the payload when it is `None` before hashing. This is what
makes the claim in the two comments above mechanically true rather than merely
intended.

**`agent/loop.py`** — adds optional `harness` and `model_version` parameters.
With `harness is None` (the default, and what any arm not using a `HarnessSpec`
gets) the three spec-consistency assertions are skipped and `runtime_version`
falls back to `f"{RUNTIME_VERSION}+{PROMPT_VERSION}"` — the *same expression*
the old code used. The fallback is not merely equivalent in spirit; it produces
the identical string, which §3 verifies on real data.

**`agent/environment.py`** — threads `StepErrorCode` through `_invalid()`. The
triggers, the observation text, the `invalid_actions` accounting and the
termination logic are untouched; the change records *why* a step was rejected,
which the old code discarded. Same control flow, richer trace.

**`tools/registry.py`** — adds a `names` property. New API, no existing caller
changes.

**`configs/harness/champion_v1.yaml`** — a new file. The `base+tool` arm does not
read it: its manifest names `configs/agent/default.yaml` and
`configs/reward/r0.yaml`, and neither changed in this range. A new config that
nothing existing loads cannot alter an existing arm's behaviour.

## 3. Mechanical check, on the stored trajectories

The audit above is a reading of the diff. The stronger claim — that the 200
stored trajectories are *unchanged objects* under the new code — is checkable
without re-running anything, because `replay.py` defines exactly what "unchanged"
means for a serialized trajectory:

```
$ uv run python -c "
import json, hashlib, orjson
from adaptive_math.agent.trace import Trajectory
from adaptive_math.agent.replay import trajectory_hash

def legacy_hash(t):
    p = t.model_dump(mode='json')
    p.pop('harness_spec_hash', None); p.pop('model_version', None)
    for e in p['events']: e.pop('error_code', None)
    return hashlib.sha256(orjson.dumps(p, option=orjson.OPT_SORT_KEYS)).hexdigest()

... over artifacts/rollout_health/thesis_e0_base_tool/trajectories.jsonl
"
trajectories loaded: 200
hash unchanged vs pre-harness serialisation: 200/200
runtime_version values seen: {'runtime-v1+agent-v1'}
```

Three things this establishes on real data, not on a hand-made example:

1. The new `Trajectory` model accepts the old records unchanged (`200/200` load
   without error), so the schema change is backward compatible in practice.
2. The three new fields really do arrive as `None` for old records, so the
   exclusion rule fires for every one of them.
3. `runtime_version` is `runtime-v1+agent-v1` for all 200 — the old-format
   string — confirming the `loop.py` fallback produces identical output.

## 4. What this establishes, and what it does not

**Establishes.** Every change on the evaluation path between the stored arm's
commit and this branch is additive: a new optional field, a new API, a type
annotation, or a new config file nothing existing loads. No control-flow
statement, no prompt, no verifier rule, no budget, and no decoding parameter was
modified. On the stored trajectories themselves, the serialized form and its
content hash are bit-identical under the new code (200/200), and the runtime tag
is unchanged. The stored `base+tool` arm is therefore pairable with arms run from
this tree.

**Does not establish.** That a *fresh* rollout under the new code would visit the
same states as one under the old code. That is a dynamic claim, and testing it
would require re-running the arm — which is the GPU cost this gate exists to
avoid spending. The argument here is a static audit plus a contract check on
recorded data; it is strong enough to justify pairing, and it is not the same
thing as having re-run the arm.

This distinction is the honest boundary of the justification, and it is the
reason the three-arm write-up should state that the `base+tool` row was recorded
at `fdf9276` under an audited-equivalent tree rather than under the identical
one.
