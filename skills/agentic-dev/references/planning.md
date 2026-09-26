# Implementation Planning and Plan Execution

Convert an approved design doc into a task-level plan, then execute it
checkbox by checkbox.

## Writing the Plan

Create a dated plan file next to the spec (e.g.
`docs/superpowers/plans/YYYY-MM-DD-<topic>.md`). Structure:

1. **Header**: goal (one paragraph), architecture (one paragraph), tech
   stack, link to the spec.
2. **Global constraints**: checkbox list of invariants that hold across all
   tasks (naming, isolation rules, budget limits, honesty rules).
3. **File structure**: the target tree, so reviewers can judge the shape
   before any code exists.
4. **Tasks**, each with:
   - Files to create/modify (exact paths)
   - Checkbox steps small enough to verify independently
   - The failing-test command to run *before* implementation
   - The verification command that must pass to close the task
   - A commit step with a conventional-commit message
5. **Exit criteria**: the measurable end state of the whole plan.

Rules:

- **Bite-sized tasks**: each task should be completable in roughly 2–5
  minutes of focused implementation. If a task needs more, split it until
  each step is trivially verifiable. The plan is written for an executor
  with no project context and no judgement — completeness beats brevity.
- **Complete content, not pointers**: include exact file paths, the
  interfaces/contracts to honor, and where cheap the actual test code or
  key snippets. The executor should never have to invent an API shape.
- Every task's first executable step is a failing test (see TDD reference).
- Plans reference only paths and APIs verified to exist or explicitly
  marked "Create:".
- Numeric thresholds in the design (sample sizes, CI levels, budgets) must
  appear verbatim in the plan — do not dilute them.
- **Parallel execution**: mark tasks with disjoint file sets as
  parallelizable; anything sharing a file is sequential by definition.
  Parallel-dispatch only tasks whose contracts cannot interfere.

## Executing the Plan

1. Work tasks in order; keep exactly one task in progress.
2. Per task: run the failing test → implement minimally → run the task's
   verification command → run the repo-wide gate (tests + lint + types).
3. Check off boxes only with command output as evidence. A checkbox without
   a passing command is a lie.
4. If reality diverges from the plan (missing dependency, wrong
   assumption), stop and amend the plan in writing before continuing; note
   the amendment and reason.
5. On completion, report: tasks done, gate output, and anything deferred —
   in the plan file's status header and to the user.

## Anti-patterns

- Batching multiple tasks before running any verification.
- "Fixing" a failing gate by weakening the test instead of the code.
- Marking the plan done while exit criteria are unmet; defer explicitly
  instead.
