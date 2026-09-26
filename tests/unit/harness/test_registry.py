"""Harness registry lifecycle: register, gate, promote, rollback."""

from pathlib import Path

import pytest

from adaptive_math.harness import HarnessRegistry, HarnessSpec
from adaptive_math.harness.gate import GatePolicy, PairedOutcome, evaluate_gate
from adaptive_math.harness.presets import CURRENT_PRODUCTION

AT = "2026-09-18T00:00:00Z"
POLICY = GatePolicy(
    policy_version="gate-v1",
    mode="non_regression",
    max_regression=0.0,
    ci_lower_bound=-0.05,
    mcnemar_alpha=0.05,
)


def _passing_gate() -> object:
    outcomes = [
        PairedOutcome(task_id=f"task:{i:03d}", champion_correct=True, candidate_correct=True)
        for i in range(20)
    ]
    return evaluate_gate(outcomes, POLICY)


def _failing_gate() -> object:
    outcomes = [
        PairedOutcome(task_id=f"task:{i:03d}", champion_correct=True, candidate_correct=False)
        for i in range(20)
    ]
    return evaluate_gate(outcomes, POLICY)


def _candidate(version: str = "h-v2") -> HarnessSpec:
    return CURRENT_PRODUCTION.model_copy(update={"harness_version": version})


def test_full_lifecycle(tmp_path: Path) -> None:
    registry = HarnessRegistry(tmp_path / "registry.json")
    champion_hash = registry.register(CURRENT_PRODUCTION, at=AT)
    registry.record_gate(champion_hash, _passing_gate(), at=AT)  # type: ignore[arg-type]
    registry.promote(champion_hash, at=AT)
    assert registry.champion is not None
    assert registry.champion.spec.harness_version == "h-v1"

    candidate_hash = registry.register(_candidate(), at=AT)
    registry.record_gate(candidate_hash, _passing_gate(), at=AT)  # type: ignore[arg-type]
    registry.promote(candidate_hash, at=AT)
    assert registry.champion is not None
    assert registry.champion.spec.harness_version == "h-v2"

    registry.rollback(champion_hash, at=AT, reason="candidate degraded in production")
    assert registry.champion is not None
    assert registry.champion.spec.harness_version == "h-v1"

    actions = [event.action for event in registry.ledger.events]
    assert actions == [
        "register", "gate", "promote",
        "register", "gate", "promote",
        "rollback",
    ]


def test_persistence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = HarnessRegistry(path)
    spec_hash = registry.register(CURRENT_PRODUCTION, at=AT)
    reloaded = HarnessRegistry(path)
    assert spec_hash in reloaded.ledger.records
    assert reloaded.ledger.records[spec_hash].spec == CURRENT_PRODUCTION


def test_register_is_idempotent(tmp_path: Path) -> None:
    registry = HarnessRegistry(tmp_path / "registry.json")
    first = registry.register(CURRENT_PRODUCTION, at=AT)
    second = registry.register(CURRENT_PRODUCTION, at=AT)
    assert first == second
    assert len(registry.ledger.records) == 1


def test_version_reuse_with_different_spec_rejected(tmp_path: Path) -> None:
    registry = HarnessRegistry(tmp_path / "registry.json")
    registry.register(CURRENT_PRODUCTION, at=AT)
    changed = CURRENT_PRODUCTION.model_copy(update={"prompt_version": "agent-v2"})
    with pytest.raises(ValueError, match="bump the version"):
        registry.register(changed, at=AT)


def test_promotion_requires_passing_gate(tmp_path: Path) -> None:
    registry = HarnessRegistry(tmp_path / "registry.json")
    spec_hash = registry.register(_candidate(), at=AT)
    with pytest.raises(ValueError, match="passing gate"):
        registry.promote(spec_hash, at=AT)
    registry.record_gate(spec_hash, _failing_gate(), at=AT)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="candidate"):
        registry.promote(spec_hash, at=AT)
    assert registry.ledger.records[spec_hash].state == "rejected"


def test_gate_only_on_candidates(tmp_path: Path) -> None:
    registry = HarnessRegistry(tmp_path / "registry.json")
    spec_hash = registry.register(CURRENT_PRODUCTION, at=AT)
    registry.record_gate(spec_hash, _passing_gate(), at=AT)  # type: ignore[arg-type]
    registry.promote(spec_hash, at=AT)
    with pytest.raises(ValueError, match="candidates"):
        registry.record_gate(spec_hash, _passing_gate(), at=AT)  # type: ignore[arg-type]


def test_rollback_requires_promoted_target(tmp_path: Path) -> None:
    registry = HarnessRegistry(tmp_path / "registry.json")
    champion_hash = registry.register(CURRENT_PRODUCTION, at=AT)
    registry.record_gate(champion_hash, _passing_gate(), at=AT)  # type: ignore[arg-type]
    registry.promote(champion_hash, at=AT)
    candidate_hash = registry.register(_candidate(), at=AT)
    with pytest.raises(ValueError, match="previously promoted"):
        registry.rollback(candidate_hash, at=AT, reason="never promoted")


def test_rollback_requires_reason(tmp_path: Path) -> None:
    registry = HarnessRegistry(tmp_path / "registry.json")
    spec_hash = registry.register(CURRENT_PRODUCTION, at=AT)
    registry.record_gate(spec_hash, _passing_gate(), at=AT)  # type: ignore[arg-type]
    registry.promote(spec_hash, at=AT)
    other = registry.register(_candidate(), at=AT)
    registry.record_gate(other, _passing_gate(), at=AT)  # type: ignore[arg-type]
    registry.promote(other, at=AT)
    with pytest.raises(ValueError, match="reason"):
        registry.rollback(spec_hash, at=AT, reason="")


def test_unknown_hash_rejected(tmp_path: Path) -> None:
    registry = HarnessRegistry(tmp_path / "registry.json")
    with pytest.raises(KeyError):
        registry.promote("0" * 64, at=AT)
