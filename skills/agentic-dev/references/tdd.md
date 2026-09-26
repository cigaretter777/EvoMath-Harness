# Test-Driven Development

Red → Green → Refactor. No production code without a failing test that
demands it.

**Hard rule:** if production code was written before its test, delete it
and start over with the test. Do not back-fill a test around existing code
and call it TDD — the test never had the chance to fail. (Declared
exceptions below are the only way out.)

## Cycle

1. **Red**: write one test for the next small behavior. Run it; watch it
   fail for the right reason (assertion on behavior, not a syntax error or
   missing import). If it fails for the wrong reason, fix the test first.
2. **Green**: write the minimum production code that passes. Hard-coding to
   pass is acceptable at this step; the next test will force generality.
3. **Refactor**: with tests green, remove duplication and clarify names.
   Run tests after every refactor step; they must stay green.

Commit at each green bar or at each task boundary, whichever the plan says.

## Test Quality Rules

- Test behavior through public interfaces, not implementation details.
- One logical assertion per test; name tests after the behavior
  (`test_rollback_requires_promoted_target`, not `test_registry_3`).
- Cover the unhappy path first for validation/parsing code: invalid input,
  boundary values, duplicates, empty inputs.
- For deterministic systems (hashing, statistics, state machines), pin
  exact expected values computed by hand or by an independent method —
  never paste the implementation's own output as the expectation without
  verifying it.
- Keep config files and code presets in sync with a bidirectional pinning
  test when the repo uses that pattern (yaml ↔ code preset).

## Pragmatic Exceptions (declare them)

- Pure wiring already covered by an integration test.
- Generated or trivially declarative code (config yaml, re-export `__init__`).
Declare the exception in the commit message or plan; do not silently skip.

## Anti-patterns

- Writing the implementation first and back-filling tests that cannot fail.
- Testing the mock instead of the system.
- A "green" run where the new test was never actually executed (check the
  collected test count before and after).
- Weakening or deleting a failing test to make the gate pass.
