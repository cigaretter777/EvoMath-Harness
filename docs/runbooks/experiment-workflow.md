# Experiment workflow runbook

Where experiment work lives, and why it lives there. Settled 2026-09-27 after
consolidating six local branches and five worktrees onto one trunk.

## One trunk: `main`

`main` is the single trunk and the only branch experiments run from. It already
carries the full history: the 2026-09-19 / 09-24 / 09-26 campaigns, the parser
tolerance + verifier fixes, and the V3 harness/evolution work. Feature branches
are short-lived and merge back before a campaign starts; campaign branches
(`campaign-*`) are retired — CI only triggers on `main` and `feature/*`, so work
parked on a `campaign-*` branch accumulates lint/type/contract debt that all
detonates at merge time (this happened on 2026-09-26).

## Run in place in the main worktree

Experiments execute in `/root/autodl-tmp/Adaptive-Solver-main-git` itself. Do
**not** create a worktree to run one. Two mechanisms bind the chain to this
directory:

- **Absolute paths.** `scripts/**` hardcodes `REPO=/root/autodl-tmp/Adaptive-Solver-main-git`
  and `PY=/root/autodl-tmp/conda-envs/adaptive-math/bin/python`; the GRPO YAMLs
  under `configs/grpo/` point at absolute model and task-pool paths.
- **`artifacts/` is gitignored.** The 75 GB of run evidence — queue journals,
  `COMPLETE` sentinels, `task_ids.txt`, shard evidence — exists only here. A
  fresh worktree starts with an empty `artifacts/`, so the idempotent queues
  cannot tell "done" from "never ran" and will re-run everything.

Changing the paths to suit a new directory means editing tracked scripts, which
makes the tree dirty, which the launch gate then rejects (next section).

The one historical exception was the H3 slot in `scripts/campaign-20260924/campaign_monitor.sh`,
which ran code from a second worktree so the E2 arms kept executing. That
worktree was retired on 2026-09-27; H3 work continues in the main worktree. The
script is kept as a record of the 09-24 campaign and is no longer runnable.

## Commit and push before every launch

`launch_grpo.sh`, `launch_sft.sh`, and `launch_r3.sh` abort when
`git status --porcelain` is non-empty, and `ADAPTIVE_MATH_ALLOW_DIRTY_RUN=1` is
deliberately never set. A run's `run_id` embeds `git rev-parse --short HEAD`, so
an uncommitted tree produces a `git_sha` describing a state that never existed —
and the run cannot be paired against any other run. That pairability is the only
reason the 2026-09-19 four-arm comparison was believable.

Keep untracked scratch out of the tree root; `.ipynb_checkpoints/` was ignored
on 2026-09-27 for exactly this reason.

## Gates and promotion

An experiment passes through, in order: the four-arm alignment gate
(`scripts/campaign-20260926/verify_eval_alignment.sh`, wired into the queue
launch path), `scripts/cloud/preflight.sh`, the contract suite, and the
dirty-tree guard. Evidence lands in `artifacts/`; numbers are promoted by hand
into `docs/results/<campaign>/report.md` (with artifact sha256) and
`stage-summary.md`, which are the tracked, citable record.

## Worktree-based development still applies to code

The worktree-first flow in `skills/agentic-dev/references/git-worktrees.md` is
fine for pure code work — V3 harness/evolution development, library changes,
tests. It does not apply to running experiments, for the two reasons above. The
rule is: **code may be written anywhere, but a run must start from a clean
committed `main` in the main worktree.**
