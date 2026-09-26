# Subagent-Driven Development

Execute a plan by dispatching one subagent per task while the main session
acts as reviewer and integrator.

**Precondition: use subagents only when the user has explicitly requested
subagent-based development.** Otherwise execute tasks in the main session.

## When It Pays

- Tasks are independent or loosely coupled (different modules, few shared
  files).
- Each task has a crisp contract: exact files, interfaces, and a
  verification command.
- The plan is already written and approved — subagents execute plans, they
  do not write them.

Do not dispatch subagents for: a single two-step change, exploratory
debugging where the hypothesis keeps changing, or tasks that all edit the
same file.

## Protocol

1. **Brief each subagent like a new colleague.** It has zero context.
   Include: the goal, the exact files to create/modify, the interfaces it
   must honor (paste the contract, not a description of it), the repo
   conventions that apply, and the exact verification command it must run
   before returning.
2. **One task per subagent.** Scope small enough that its result can be
   reviewed in minutes.
3. **Review every result in two stages** (see code-review.md):
   - *Stage 1 — spec compliance*: the diff implements exactly the task,
     nothing missing, nothing unrequested. A spec failure here means the
     code goes back regardless of how elegant it is.
   - *Stage 2 — code quality*: correctness, convention consistency, test
     quality, simplicity. Only after Stage 1 passes.
   Read the diff yourself and run the task's verification command plus the
   repo-wide gate. A subagent's summary is a claim, not evidence.
4. **Integrate sequentially.** Merge one task's changes, re-run the gate,
   then dispatch the next dependent task. Parallel-dispatch only tasks with
   disjoint file sets (see the parallel-agents guidance in planning.md).
5. **Record outcomes in the plan**: check off with the gate output you
   observed, not the subagent's report.

## Failure Handling

- Subagent fails its own verification: send the failure output back to the
  same subagent (resume if possible) with instruction to fix; do not patch
  it silently in the main session unless the fix is trivial.
- Repeated failure (2+ rounds): take the task back into the main session
  and note the substitution in the plan.
- Subagent violates scope (touches files outside its task): revert those
  edits before integrating.

## Anti-patterns

- Dispatching without a written plan (subagents improvising architecture).
- Integrating unreviewed code because the summary sounded confident.
- Running two subagents on overlapping files and resolving conflicts by
  hand afterwards.
