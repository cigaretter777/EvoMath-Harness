# Using Git Worktrees

Isolate each approved plan in its own worktree and branch so concurrent
work and experiments never contaminate the main checkout.

## Setup (after design approval, before Task 1)

1. Confirm the repo is a git repository with a clean-ish status; commit or
   stash unrelated changes first.
2. Create the worktree on a new branch named after the plan:
   `git worktree add ../<repo>-<plan-slug> -b <plan-slug>`
3. Run the project's setup in the worktree (dependency sync, env files —
   copy what is gitignored but required).
4. **Verify a clean test baseline** in the worktree before writing any
   code: the full suite must pass, or every pre-existing failure must be
   documented in the plan so it is never attributed to your changes.

## During Development

- All task commits happen on the worktree branch; the main checkout stays
  untouched and runnable.
- Never commit generated artifacts, secrets, or large binaries; check
  `git status` before every commit.
- Keep commits at green bars or task boundaries with conventional-commit
  messages referencing the plan task.

## Anti-patterns

- Starting Task 1 without a documented clean baseline.
- Editing files in both the main checkout and the worktree for the same
  plan.
- Letting the worktree drift far behind main without a deliberate rebase
  decision recorded in the plan.
