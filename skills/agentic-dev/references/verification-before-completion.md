# Verification Before Completion

Run this checklist before claiming any task, fix, or plan is done. A claim
without fresh command output is a hypothesis, not a result.

## The Checklist

1. **Fresh evidence.** Run the verification command *now*, in this turn —
   do not cite output from earlier in the session; the code may have
   changed since.
2. **The thing actually ran.** Check collected/passed test counts, exit
   codes, and output paths. A typo'd test path that collects 0 tests is a
   green lie.
3. **Whole gate, not just the new part.** Full test suite + lint + type
   check (use the repo's documented commands). New code passing while old
   tests break is not done.
4. **Experience the deliverable.** For artifacts (apps, pages, documents,
   charts), open or render the output the way its recipient will; a file
   that exists but was never opened is unverified.
5. **Numbers reconcile.** Totals equal the sum of line items; derived
   values match their inputs; quoted numbers trace to the artifact that
   produced them.
6. **Scope audit.** `git status` / diff shows only what the task intended —
   no stray edits, debug scaffolding, or unrelated "improvements".

## Reporting Honestly

Report in four buckets, never blur them:

- **Done** — with the command output that proves it.
- **Attempted but failed** — with the failure output.
- **Not verifiable** — say what could not be checked and why.
- **Deferred** — with the reason and where it is tracked.

## Anti-patterns

- "All tests pass" cited from memory or an earlier run.
- Marking checkboxes done in batches at the end.
- Treating "the script ran without crashing" as "the output is correct".
- Weakening a check (deleting a test, loosening an assertion) to reach green.
