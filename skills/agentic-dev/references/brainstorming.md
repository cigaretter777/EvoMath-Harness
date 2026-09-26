# Brainstorming-Driven Design

Turn a vague feature idea into one approved, written design. Diverge first,
converge second; never present a single solution as if it were the only one.

## Process

1. **Ground in reality.** Read the actual code, configs, and docs the design
   will touch. List the existing conventions the design must reuse (types,
   versioning, hashing, persistence, test patterns). Do not design from the
   user's description alone.

2. **Clarify with questions, one round at a time.** Ask only what changes
   the design: scope boundaries, constraints, success criteria. Prefer
   offering 2–3 concrete interpretations the user can pick with one word
   over open-ended questions.

3. **Diverge: generate 2–3 genuinely different approaches.** For each:
   one-paragraph mechanism, what it reuses, what it costs, its main risk.
   Different means structurally different (e.g. data-model change vs.
   process change vs. new component), not three namings of one idea.

4. **Converge: recommend one, with reasons.** State the trade-off that
   decided it and what would change your mind. Invite the user to override.

5. **Write the design doc.** Dated file in the repo's docs tree
   (e.g. `docs/superpowers/specs/YYYY-MM-DD-<topic>.md`). Required sections:
   - Executive summary with the core loop/architecture in one diagram
   - Problem and motivation, tied to observed code or prior incidents
   - Definitions and boundaries (what is in scope, what is explicitly not)
   - Data contracts (schemas, invariants, versioning and hashing rules)
   - Per-component design with integration points into existing modules
   - Non-goals and "must not claim" list (prevents overclaiming later)
   - Phased implementation with acceptance gates per phase
   - Risks and mitigations
   - Current implementation status, honestly separated: built vs. planned

6. **Get explicit approval** before any planning or code. "Looks good"
   from the user is the gate; silence is not.

## Anti-patterns

- Designing around conventions you have not read (invented file paths,
  invented APIs).
- A design doc that only describes the happy path — no failure modes, no
  rollback story.
- Mixing "what exists" with "what is planned" in one status claim.
- Scope creep mid-design: park new ideas in a "future work" section instead.
