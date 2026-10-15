"""Anchor pool rules (report sections 6.1 and 10.6)."""

import asyncio

from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.loop import AgentLoop, ConversationJournal
from adaptive_math.agent.model_client import GenerationConfig
from adaptive_math.core.types import AnswerType, Budget, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.recovery.anchors import (
    MIN_REMAINING_PYTHON_SECONDS,
    MIN_REMAINING_STEPS,
    MIN_REMAINING_TOOL_CALLS,
    build_candidate,
    first_exec_error_snapshot,
    pool_verdict,
    select_one_per_task,
    summarise_candidates,
)
from adaptive_math.tools.registry import ToolRegistry

from .fakes import DeadSandboxTool, FakePythonTool, ScriptedModel, SlowTool, final, tool_call

GENERATION = GenerationConfig(max_new_tokens=256, temperature=0.7)


def _budget(steps: int = 8, calls: int = 4, python_seconds: float = 12.0) -> Budget:
    return Budget(
        max_steps=steps,
        max_tool_calls=calls,
        max_python_seconds=python_seconds,
        max_observation_chars=8000,
    )


def _labeled(task_id: str = "unit:anchor") -> LabeledMathTask:
    return LabeledMathTask(
        task=MathTask(
            task_id=task_id,
            problem="Compute the value.",
            answer_type=AnswerType.EXPRESSION,
            dataset="unit",
            split="frozen_eval",
            source_hash="b" * 64,
            pipeline_version="unit-v1",
        ),
        reference=ReferenceAnswer(value="42", answer_type=AnswerType.EXPRESSION),
    )


def _journal(
    script: list[str],
    *,
    budget: Budget,
    registry: ToolRegistry,
    task_id: str = "unit:anchor",
) -> ConversationJournal:
    journal = ConversationJournal()
    environment = OfflineMathEnv(_labeled(task_id), budget, registry, trace_id="t")
    asyncio.run(AgentLoop().run(environment, ScriptedModel(script), GENERATION, journal=journal))
    return journal


def _candidates(journal: ConversationJournal, *, sample_index: int = 0, task_id: str = "unit:anchor"):
    snapshot = first_exec_error_snapshot(journal.snapshots)
    if snapshot is None:
        return None
    return build_candidate(
        snapshot=snapshot, task_id=task_id, source_trajectory_id="t", sample_index=sample_index
    )


def test_the_anchor_is_the_first_execution_error_only() -> None:
    """A later error must not displace the first one: the pool is defined on
    the trajectory's first executed failure."""
    budget = _budget()
    registry = ToolRegistry([FakePythonTool()])
    journal = _journal(
        [tool_call("BOOM1"), tool_call("BOOM2"), final("42")], budget=budget, registry=registry
    )

    snapshot = first_exec_error_snapshot(journal.snapshots)
    assert snapshot is not None
    assert snapshot.step_index == 0
    assert "ZeroDivisionError" in snapshot.messages[-1].content


def test_a_timeout_is_not_an_execution_error() -> None:
    """E_TIMEOUT stays out of the C pool (section 6)."""
    journal = _journal(
        [tool_call("while True: pass"), final("42")],
        budget=_budget(),
        registry=ToolRegistry([SlowTool()]),
    )

    assert first_exec_error_snapshot(journal.snapshots) is None


def test_infrastructure_faults_produce_no_anchor() -> None:
    journal = _journal(
        [tool_call("print(1)"), final("42")],
        budget=_budget(),
        registry=ToolRegistry([DeadSandboxTool()]),
    )

    assert first_exec_error_snapshot(journal.snapshots) is None


def test_an_anchor_needs_steps_calls_and_python_time_left() -> None:
    budget = _budget(steps=MIN_REMAINING_STEPS, calls=MIN_REMAINING_TOOL_CALLS)
    registry = ToolRegistry([FakePythonTool()])
    journal = _journal([tool_call("BOOM"), final("42")], budget=budget, registry=registry)
    candidate = _candidates(journal)

    assert candidate is not None
    # Exactly at the floor: 8 steps budget would have been fine, 3 is the line.
    assert candidate.remaining_steps == MIN_REMAINING_STEPS - 1
    assert candidate.eligible is False
    assert candidate.exclusion_reason == "insufficient_remaining_steps"


def test_remaining_python_time_is_read_from_spent_seconds() -> None:
    budget = _budget(python_seconds=MIN_REMAINING_PYTHON_SECONDS + 0.1)
    journal = _journal(
        [tool_call("BOOM"), final("42")],
        budget=budget,
        registry=ToolRegistry([FakePythonTool(execution_time=0.25)]),
    )
    candidate = _candidates(journal)

    assert candidate is not None
    assert candidate.remaining_python_seconds < MIN_REMAINING_PYTHON_SECONDS
    assert candidate.exclusion_reason == "insufficient_remaining_python_seconds"


def test_a_terminated_anchor_is_excluded() -> None:
    """The failing call is the last allowed step, so nothing can follow it."""
    journal = _journal(
        [tool_call("BOOM"), final("42")],
        budget=_budget(steps=1, calls=4),
        registry=ToolRegistry([FakePythonTool()]),
    )
    candidate = _candidates(journal)

    assert candidate is not None
    assert candidate.exclusion_reason == "terminated_at_anchor"
    assert candidate.eligible is False


def test_one_anchor_per_task_chosen_by_stable_hash() -> None:
    budget = _budget()
    registry = ToolRegistry([FakePythonTool()])
    candidates = []
    for sample_index in range(4):
        journal = _journal(
            [tool_call("BOOM"), final("42")], budget=budget, registry=registry
        )
        candidate = _candidates(journal, sample_index=sample_index)
        assert candidate is not None
        # Trajectory identity is what the ordering key must distinguish.
        candidate = build_candidate(
            snapshot=candidate.snapshot,
            task_id="unit:anchor",
            source_trajectory_id=f"t{sample_index}",
            sample_index=sample_index,
        )
        candidates.append(candidate)

    selected = select_one_per_task(candidates)

    assert len(selected) == 1
    # Order is a pure function of identity, so it is reproducible.
    assert selected == select_one_per_task(list(reversed(candidates)))
    assert selected[0].task_id == "unit:anchor"


def test_exclusion_reasons_are_counted_not_dropped() -> None:
    journal = _journal(
        [tool_call("BOOM"), final("42")],
        budget=_budget(steps=1, calls=4),
        registry=ToolRegistry([FakePythonTool()]),
    )
    candidate = _candidates(journal)
    assert candidate is not None

    summary = summarise_candidates([candidate])

    assert summary["candidates"] == 1
    assert summary["eligible"] == 0
    assert summary["excluded:terminated_at_anchor"] == 1


def test_pool_verdict_thresholds_match_the_report() -> None:
    assert pool_verdict(40) == "C"
    assert pool_verdict(20) == "C"
    assert pool_verdict(19) == "C_SMALL_SAMPLE"
    assert pool_verdict(8) == "C_SMALL_SAMPLE"
    assert pool_verdict(7) == "INSUFFICIENT_ERROR_STATES"


def test_candidate_records_carry_an_auditable_public_view() -> None:
    journal = _journal(
        [tool_call("BOOM"), final("42")],
        budget=_budget(),
        registry=ToolRegistry([FakePythonTool()]),
    )
    candidate = _candidates(journal)
    assert candidate is not None

    view = candidate.public_view()

    assert view["anchor_id"] == "unit:anchor::s0::step0"
    assert view["error_class"] == "E_EXEC"
    assert view["prefix_event_count"] == 3
    assert view["prefix_message_count"] == 4
    assert len(str(view["state_hash"])) == 64
    assert "snapshot" not in view
    assert "ZeroDivisionError" in str(view["observation"])
