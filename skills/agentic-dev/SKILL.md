---
name: agentic-dev
description: Agentic software development methodology — brainstorming-driven design (头脑风暴/方案设计), implementation planning and executing-plans (任务计划), test-driven development (TDD/红绿重构), subagent-driven development with two-stage review (子代理开发), systematic debugging (系统化调试/根因分析), verification-before-completion (完成前验证), code review (代码审查), git worktrees, and branch finishing. Use when the user asks to design a feature or architecture, brainstorm approaches, write or execute an implementation plan, practice TDD, debug systematically, review code, develop with subagents, or when a repo document references superpowers-style workflows (brainstorming / writing-plans / executing-plans / tdd / subagent-driven-development / systematic-debugging / verification-before-completion / requesting-code-review / using-git-worktrees / finishing-a-development-branch).
---

# Agentic Development Methodology

Composable workflows. Route by user intent; read only the references the
current intent needs.

## Routing

| Intent | Reference to read |
|---|---|
| Design a feature, explore approaches, "头脑风暴", architecture decision | [references/brainstorming.md](references/brainstorming.md) |
| Turn an approved design into a task-level plan, or execute an existing plan | [references/planning.md](references/planning.md) |
| Write or modify production code | [references/tdd.md](references/tdd.md) |
| User explicitly asks for subagent-based development | [references/subagent-driven-development.md](references/subagent-driven-development.md) |
| Bug, failing test, or unexpected behavior | [references/systematic-debugging.md](references/systematic-debugging.md) |
| Between tasks; reviewing a diff; responding to review feedback | [references/code-review.md](references/code-review.md) |
| Starting an approved plan in a git repo | [references/git-worktrees.md](references/git-worktrees.md) |
| All tasks complete; merge/PR/keep/discard decision | [references/finishing-branch.md](references/finishing-branch.md) |
| Before claiming anything is done (always) | [references/verification-before-completion.md](references/verification-before-completion.md) |

## The Full Cycle

1. **Brainstorm** — converge on one design, written to a dated spec file, approved by the user.
2. **Worktree** — isolated branch, project setup, documented clean test baseline.
3. **Plan** — bite-sized tasks (2–5 min each) with exact files, contracts, and verification commands.
4. **Execute** — task-by-task with TDD; subagents only when the user explicitly asks.
5. **Review** — two-stage review (spec compliance, then code quality) between tasks; Critical issues block.
6. **Finish** — final gate, exit-criteria audit, user chooses merge/PR/keep/discard, cleanup.

## Iron Rules

- Never write production code before a design is approved in writing.
- Never write production code without a failing test first; delete code written before its test.
- Never mark a task done while its verification command fails; cite fresh command output as evidence.
- Match the host repository's existing conventions (types, hashing, config
  patterns, test layout) over personal preference; read the neighboring
  modules before designing anything.
- Design docs and plans live in the repo's existing docs structure, dated
  (YYYY-MM-DD), and reference real file paths — not invented ones.
- Use subagents only when the user explicitly requests them.
