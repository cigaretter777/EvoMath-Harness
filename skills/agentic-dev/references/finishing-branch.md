# Finishing a Development Branch

When all plan tasks are complete, close out deliberately — do not just
leave a branch lying around.

## Pre-flight

1. Run the full gate one final time in the worktree: whole test suite +
   lint + types, fresh output.
2. Re-check the plan's exit criteria one by one, with evidence for each.
3. Audit the branch diff as a whole (not task by task): scope, no
   scaffolding, no secrets, docs updated to match final behavior.

## Present the Decision to the User

Offer exactly these options with a recommendation:

- **Merge** — fast-forward or merge-commit into the base branch, then
  delete the feature branch.
- **Pull request** — push and open a PR with a summary generated from the
  plan (goal, tasks, gate evidence, deviations).
- **Keep** — leave the branch as-is, with a stated reason (e.g. awaiting
  review, experimental).
- **Discard** — the work failed validation; document why in the plan
  before removing anything. Deleting work requires explicit user approval.

## Cleanup (after merge/PR/discard)

- Remove the worktree: `git worktree remove ../<repo>-<plan-slug>`.
- Prune the branch if merged or discarded.
- Update the plan file's status header: final state, gate output, links to
  merge/PR.
- Surface follow-ups discovered along the way as new tickets or a "future
  work" section — never as silent scope creep into the next plan.

## Anti-patterns

- Merging with a red or stale gate run.
- Summarizing the branch from memory instead of from the plan and diff.
- Deleting branches or worktrees without explicit user approval.
