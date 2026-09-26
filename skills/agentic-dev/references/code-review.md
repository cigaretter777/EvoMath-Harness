# Code Review: Requesting and Receiving

Review happens between tasks, not only at the end. Critical issues block
progress; they are never deferred silently.

## Requesting a Review (self-review or reviewer subagent)

Run this after each task's gate passes and before the next task starts.

**Stage 1 — Spec compliance:**
- Does the diff implement exactly what the task specified — no more, no
  less? List any missing requirement and any unrequested addition.
- Are the task's file list and interfaces honored?

**Stage 2 — Code quality:**
- Correctness: edge cases, error paths, off-by-one, resource cleanup.
- Consistency: does it follow the repo's existing conventions (types,
  validation, hashing, persistence, naming) rather than inventing parallel
  patterns?
- Tests: do they test behavior through public interfaces? Can any test
  pass while the code is wrong? Is the unhappy path covered?
- Simplicity: is there a simpler shape that satisfies the same tests
  (YAGNI, DRY)?

**Report by severity:**
- **Critical** — wrong behavior, data loss risk, broken invariant, missing
  requirement. Blocks progress; fix before continuing.
- **Major** — convention violation, weak test, unnecessary complexity.
  Fix in the current task or explicitly ticket with user visibility.
- **Minor** — naming, comments, style. Fix opportunistically.

## Receiving a Review

- Treat every finding as information, not criticism. Restate the finding in
  your own words before responding to it.
- Fix Critical and Major findings with TDD: write the test that exposes the
  issue, watch it fail, fix, watch it pass.
- If you disagree with a finding, say why with evidence (a test, a doc, a
  convention citation) — never silently ignore it.
- After fixes, re-run the whole gate and report what changed.

## Anti-patterns

- Reviewing only the happy path and skipping error handling.
- Approving because "tests pass" without reading the diff.
- Arguing with findings instead of reproducing them.
- Deferring Critical issues to "later" without telling the user.
