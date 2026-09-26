# Systematic Debugging

Four-phase root-cause process. Never guess at fixes; every fix must trace
back to a root cause you can state in one sentence.

## Phase 1: Reproduce and Stabilize

- Build the smallest reliable reproduction: a failing test, a script, or a
  deterministic command. Intermittent failures get a loop (run N times,
  record the failure rate) before any diagnosis.
- Capture the exact failure signature: full error text, stack, inputs,
  environment. Save it — later phases compare against it.
- If the failure cannot be reproduced, the task is instrumentation, not
  fixing. Add logging/assertions at suspected boundaries until it can.

## Phase 2: Trace the Root Cause

- Work backwards from the failure signature along the data flow. At each
  boundary ask: "is the value wrong here, or was it already wrong when it
  arrived?" Bisect until you find the first wrong point.
- Distinguish the **root cause** (the first wrong thing) from **symptoms**
  (everything downstream). Fixing a symptom leaves the cause alive.
- State the root cause in one sentence: "X happens because Y under
  condition Z." If you cannot, you are not done with this phase.
- For timing/race issues: never fix by adding arbitrary sleeps. Wait on a
  concrete condition (a state, an event, a file) with a timeout and a
  useful error message.

## Phase 3: Fix at the Root, with Defense in Depth

- Write the failing regression test first (TDD red) that encodes the root
  cause condition.
- Fix at the root-cause layer, not at every symptom site.
- Ask what *class* of bug this is and whether the same class exists
  elsewhere; fix or ticket those instances too.
- Add one layer of defense where cheap: a validation, an invariant
  assertion, or a type narrowing that makes this bug class impossible or
  immediately loud.

## Phase 4: Verify Completely

- The regression test passes; the original reproduction command passes.
- Run the full gate (whole test suite + lint + types), not just the new
  test — root-cause fixes can have wide blast radius.
- Confirm no diagnostic scaffolding is left behind (debug prints, temporary
  sleeps, commented-out checks).
- Then follow verification-before-completion before declaring the fix done.

## Anti-patterns

- "Try something and see" — changing code before Phase 2 is complete.
- Fixing the symptom closest to the error message.
- Retry loops and sleeps as fixes for races.
- Declaring fixed because the error message changed.
