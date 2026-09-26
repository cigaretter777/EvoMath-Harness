"""Append-only harness registry with promotion and rollback.

The registry is a single JSON ledger: every known ``HarnessSpec`` is stored
under its content hash with a lifecycle state, and a separate ``champion``
pointer names the harness currently serving rollouts. Promotion requires a
passing ``GateDecision`` recorded for the candidate; rollback moves the
champion pointer back to any previously promoted hash. Entries are never
deleted and decisions are never edited — new information arrives as new
events, mirroring the journal/artifact patterns in ``evaluation.model_eval``.
"""

import json
import os
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from adaptive_math.harness.gate import GateDecision
from adaptive_math.harness.spec import HarnessSpec

REGISTRY_SCHEMA_VERSION = "harness-registry-v1"

HarnessState = Literal["candidate", "promoted", "rejected"]


class RegistryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["register", "gate", "promote", "reject", "rollback"]
    spec_hash: str
    at: str
    detail: str = ""


class HarnessRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spec: HarnessSpec
    state: HarnessState
    registered_at: str
    gate: GateDecision | None = None


class RegistryLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = REGISTRY_SCHEMA_VERSION
    champion: str | None = None
    records: dict[str, HarnessRecord] = Field(default_factory=dict)
    events: tuple[RegistryEvent, ...] = ()


class HarnessRegistry:
    """File-backed ledger; every mutation is one atomic JSON replace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            self._ledger = RegistryLedger.model_validate_json(path.read_text())
        else:
            self._ledger = RegistryLedger()
            self._persist()

    @property
    def ledger(self) -> RegistryLedger:
        return self._ledger

    @property
    def champion(self) -> HarnessRecord | None:
        if self._ledger.champion is None:
            return None
        return self._ledger.records[self._ledger.champion]

    def register(self, spec: HarnessSpec, *, at: str) -> str:
        """Add a candidate; returns its content hash. Idempotent per hash."""
        spec_hash = spec.spec_hash()
        existing = self._ledger.records.get(spec_hash)
        if existing is not None:
            return spec_hash
        for other_hash, record in self._ledger.records.items():
            if record.spec.harness_version == spec.harness_version and other_hash != spec_hash:
                raise ValueError(
                    f"harness_version {spec.harness_version!r} already registered with "
                    f"a different spec hash; bump the version for a new patch"
                )
        record = HarnessRecord(spec=spec, state="candidate", registered_at=at)
        self._mutate(spec_hash, record, RegistryEvent(action="register", spec_hash=spec_hash, at=at))
        return spec_hash

    def record_gate(self, spec_hash: str, decision: GateDecision, *, at: str) -> None:
        record = self._require(spec_hash)
        if record.state != "candidate":
            raise ValueError(f"gate decisions only apply to candidates, not {record.state}")
        updated = record.model_copy(update={"gate": decision})
        action: Literal["gate", "reject"] = "gate" if decision.passed else "reject"
        state: HarnessState = "candidate" if decision.passed else "rejected"
        updated = updated.model_copy(update={"state": state})
        self._mutate(
            spec_hash,
            updated,
            RegistryEvent(
                action=action, spec_hash=spec_hash, at=at,
                detail=f"passed={decision.passed} delta={decision.accuracy_delta:.4f}",
            ),
        )

    def promote(self, spec_hash: str, *, at: str) -> None:
        record = self._require(spec_hash)
        if record.state != "candidate":
            raise ValueError(f"only a candidate can be promoted, not {record.state}")
        if record.gate is None or not record.gate.passed:
            raise ValueError("promotion requires a recorded passing gate decision")
        previous = self._ledger.champion
        if previous is not None:
            prev_record = self._ledger.records[previous]
            self._ledger = self._ledger.model_copy(
                update={"records": {
                    **self._ledger.records,
                    previous: prev_record.model_copy(update={"state": "promoted"}),
                }}
            )
        promoted = record.model_copy(update={"state": "promoted"})
        self._mutate(
            spec_hash,
            promoted,
            RegistryEvent(
                action="promote", spec_hash=spec_hash, at=at,
                detail=f"previous_champion={previous}",
            ),
            champion=spec_hash,
        )

    def rollback(self, target_hash: str, *, at: str, reason: str) -> None:
        """Move the champion pointer back to a previously promoted harness."""
        if not reason:
            raise ValueError("rollback requires an explicit reason")
        target = self._require(target_hash)
        if target.state != "promoted":
            raise ValueError("rollback target must be a previously promoted harness")
        if self._ledger.champion is None:
            raise ValueError("no champion to roll back from")
        if self._ledger.champion == target_hash:
            raise ValueError("target is already the champion")
        current = self._ledger.champion
        self._mutate(
            target_hash,
            target,
            RegistryEvent(
                action="rollback", spec_hash=target_hash, at=at,
                detail=f"from={current} reason={reason}",
            ),
            champion=target_hash,
        )

    def _require(self, spec_hash: str) -> HarnessRecord:
        record = self._ledger.records.get(spec_hash)
        if record is None:
            raise KeyError(f"unknown harness spec hash: {spec_hash}")
        return record

    def _mutate(
        self,
        spec_hash: str,
        record: HarnessRecord,
        event: RegistryEvent,
        *,
        champion: str | None | object = ...,  # sentinel: keep current champion
    ) -> None:
        update: dict[str, object] = {
            "records": {**self._ledger.records, spec_hash: record},
            "events": (*self._ledger.events, event),
        }
        if champion is not ...:
            update["champion"] = cast(str | None, champion)
        self._ledger = self._ledger.model_copy(update=update)
        self._persist()

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rendered = self._ledger.model_dump_json(indent=2) + "\n"
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(rendered)
        os.replace(temporary, self.path)


def load_ledger(path: Path) -> RegistryLedger:
    """Read-only view used by report/audit tooling."""
    return RegistryLedger.model_validate_json(json.loads(path.read_text()))
