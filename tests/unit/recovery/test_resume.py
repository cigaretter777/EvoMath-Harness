"""Report section 10's acceptance tests for prefix-resume continuations.

The equality tests deliberately exclude ``monotonic_ms``: it records real
elapsed time in whichever process ran the branch, so a resumed suffix can never
reproduce the original wall clock. Ordering across the seam is asserted
separately, because that is the property that must hold.
"""

import asyncio
import json

import pytest

from adaptive_math.agent.actions import FinalAction
from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.loop import AgentLoop, ConversationJournal
from adaptive_math.agent.model_client import GenerationConfig
from adaptive_math.agent.prompts import render_initial_messages
from adaptive_math.agent.state import TerminationReason
from adaptive_math.core.types import AnswerType, Budget, LabeledMathTask, MathTask, ReferenceAnswer
from adaptive_math.recovery.anchors import AnchorCandidate, build_candidate, state_hash
from adaptive_math.recovery.arms import HINT_TEXT, MARKER_TEXT, Arm
from adaptive_math.recovery.resume import build_prefix_messages, run_continuation
from adaptive_math.recovery.seeds import generation_seed
from adaptive_math.tools.registry import ToolRegistry

from .fakes import DeadSandboxTool, FakePythonTool, ScriptedModel, final, tool_call

REFERENCE = "1234567/89"
GENERATION = GenerationConfig(max_new_tokens=256, temperature=0.7)


def _budget(
    *, steps: int = 6, calls: int = 4, python_seconds: float = 12.0
) -> Budget:
    return Budget(
        max_steps=steps,
        max_tool_calls=calls,
        max_python_seconds=python_seconds,
        max_observation_chars=8000,
    )


def _labeled(*, reference: str = REFERENCE) -> LabeledMathTask:
    task = MathTask(
        task_id="unit:recovery",
        problem="Compute the value.",
        answer_type=AnswerType.EXPRESSION,
        dataset="unit",
        split="frozen_eval",
        source_hash="a" * 64,
        pipeline_version="unit-v1",
    )
    return LabeledMathTask(
        task=task,
        reference=ReferenceAnswer(value=reference, answer_type=AnswerType.EXPRESSION),
    )


SCRIPT = [tool_call("BOOM"), tool_call("print(6*7)"), final("42")]


def _run_natural(
    *,
    budget: Budget,
    registry: ToolRegistry,
    script: list[str] | None = None,
    labeled: LabeledMathTask | None = None,
):
    labeled = labeled or _labeled()
    model = ScriptedModel(script or list(SCRIPT))
    journal = ConversationJournal()
    environment = OfflineMathEnv(labeled, budget, registry, trace_id="natural")
    agent = AgentLoop()
    trajectory = asyncio.run(
        agent.run(environment, model, GENERATION, journal=journal)
    )
    return environment, model, journal, trajectory


def _anchor_from(journal: ConversationJournal, *, task_id: str = "unit:recovery") -> AnchorCandidate:
    snapshot = journal.snapshots[0]
    return build_candidate(
        snapshot=snapshot,
        task_id=task_id,
        source_trajectory_id="natural",
        sample_index=0,
    )


def _events_shape(trajectory) -> list[tuple[int, str, str, str | None]]:
    return [
        (event.sequence, event.kind.value, json.dumps(event.payload, sort_keys=True), event.error_code)
        for event in trajectory.events
    ]


def _continue(
    anchor: AnchorCandidate,
    *,
    arm: Arm,
    budget: Budget,
    registry: ToolRegistry,
    script: list[str] | None = None,
    labeled: LabeledMathTask | None = None,
    sample_index: int = 0,
    attempt: int = 0,
):
    model = ScriptedModel(script or list(SCRIPT), model_id="scripted")
    agent = AgentLoop()
    return model, asyncio.run(
        run_continuation(
            run_id="fr-test",
            stage="C",
            anchor=anchor,
            labeled_task=labeled or _labeled(),
            arm=arm,
            sample_index=sample_index,
            attempt=attempt,
            model=model,
            agent=agent,
            generation=GENERATION,
            registry=registry,
            model_version="scripted",
            protocol_hash="protocol-test",
        )
    )


def test_resume_reproduces_the_natural_suffix_exactly() -> None:
    """Section 10.1: a breakpoint resume must produce the same suffix events,
    usage and final result as the uninterrupted run."""
    budget = _budget()
    registry = ToolRegistry([FakePythonTool()])
    labeled = _labeled(reference="42")
    environment, _model, journal, natural = _run_natural(
        budget=budget, registry=registry, labeled=labeled
    )
    anchor = _anchor_from(journal)
    assert anchor.eligible

    _branch_model, outcome = _continue(
        anchor, arm=Arm.FULL, budget=budget, registry=registry, labeled=labeled
    )

    prefix_count = anchor.prefix_event_count
    assert _events_shape(outcome.trajectory)[prefix_count:] == _events_shape(natural)[prefix_count:]
    assert outcome.trajectory.usage == natural.usage
    assert outcome.trajectory.final_answer == natural.final_answer == "42"
    assert outcome.trajectory.termination_reason is TerminationReason.FINAL
    # The hidden verifier is built on the resume path too, so the branch is
    # judged by the same reference the natural run was.
    assert environment.evaluate() is not None
    assert outcome.record["correct"] is True
    assert outcome.record["suffix_usage"] == {
        "steps": 2,
        "tool_calls": 1,
        "python_seconds": pytest.approx(0.25),
        "generated_tokens": 14,
        "invalid_actions": 0,
    }


def test_resumed_events_stay_monotonic_across_the_seam() -> None:
    budget = _budget()
    registry = ToolRegistry([FakePythonTool()])
    _environment, _model, journal, _natural = _run_natural(budget=budget, registry=registry)
    anchor = _anchor_from(journal)

    _branch_model, outcome = _continue(anchor, arm=Arm.FULL, budget=budget, registry=registry)

    times = [event.monotonic_ms for event in outcome.trajectory.events]
    assert times == sorted(times)
    assert times[anchor.prefix_event_count] >= times[anchor.prefix_event_count - 1]


def test_resume_never_resets_the_budget() -> None:
    """Section 10.2: the branch must inherit spent steps, calls and python time.

    The natural run spends 2 of the 2 allowed calls; a branch resumed from
    after its first call may spend one more and is then refused, which is only
    observable if the counter was inherited rather than reset."""
    budget = _budget(steps=5, calls=2)
    registry = ToolRegistry([FakePythonTool()])
    _environment, _model, journal, natural = _run_natural(
        budget=budget, registry=registry, script=[tool_call("BOOM"), tool_call("BOOM"), final("0")]
    )
    first = _anchor_from(journal)
    assert first.remaining_tool_calls == 1
    assert not first.eligible  # 1 remaining call < the pool minimum of 2

    _branch_model, outcome = _continue(
        first,
        arm=Arm.FULL,
        budget=budget,
        registry=registry,
        script=[tool_call("BOOM"), tool_call("print(1)"), tool_call("print(2)"), final("0")],
    )

    record = outcome.record
    assert record["remaining_budget_at_anchor"]["tool_calls"] == 1
    assert record["remaining_budget_at_anchor"]["steps"] == 4
    suffix = [event.kind.value for event in outcome.trajectory.events[first.prefix_event_count :]]
    assert suffix == [
        "model_output",
        "tool_call",
        "tool_result",
        "model_output",
        "invalid_action",
        "model_output",
        "final",
    ]
    # The third call was refused, so the counter stops where the budget says.
    assert outcome.trajectory.usage.tool_calls == natural.usage.tool_calls == 2
    assert "Tool-call budget exhausted" in json.dumps(
        outcome.record["suffix_events"][4]["payload"]
    )


def test_arms_differ_only_in_the_anchor_observation() -> None:
    """Section 10.4: F must be the frozen text, and the three prefixes must be
    identical everywhere except the single intervened message."""
    budget = _budget()
    registry = ToolRegistry([FakePythonTool()])
    _environment, _model, journal, _natural = _run_natural(budget=budget, registry=registry)
    snapshot = journal.snapshots[0]
    anchor = _anchor_from(journal)

    rendered = {
        arm: build_prefix_messages(snapshot, arm) for arm in (Arm.FULL, Arm.PROMPTED, Arm.MARKER)
    }
    full_messages, original, same = rendered[Arm.FULL]
    assert same == original == anchor.observation
    prompted_messages, _o, prompted = rendered[Arm.PROMPTED]
    marker_messages, _o, marker = rendered[Arm.MARKER]

    assert prompted == f"{original}\n{HINT_TEXT}"
    assert marker == MARKER_TEXT
    for messages in (prompted_messages, marker_messages):
        assert [message.role for message in messages] == [message.role for message in full_messages]
        assert messages[:-1] == full_messages[:-1]
        assert messages[-1].content != full_messages[-1].content
    # The frozen text carries the interpreter's own words, which is what arm M
    # (and only arm M) drops.
    assert "ZeroDivisionError" in original


def test_every_arm_inherits_the_same_state_hash_and_budget() -> None:
    budget = _budget()
    registry = ToolRegistry([FakePythonTool()])
    _environment, _model, journal, _natural = _run_natural(budget=budget, registry=registry)
    anchor = _anchor_from(journal)

    records = {}
    for arm in (Arm.FULL, Arm.PROMPTED, Arm.MARKER):
        _branch_model, outcome = _continue(anchor, arm=arm, budget=budget, registry=registry)
        records[arm] = outcome.record

    assert {record["state_hash"] for record in records.values()} == {anchor.state_hash}
    assert {json.dumps(record["remaining_budget_at_anchor"]) for record in records.values()} == {
        json.dumps(records[Arm.FULL]["remaining_budget_at_anchor"])
    }
    # Every branch starts from the same prefix and the same first tool call.
    assert {record["prefix_event_count"] for record in records.values()} == {
        anchor.prefix_event_count
    }
    assert {
        record["suffix_events"][0]["kind"] for record in records.values()
    } == {"model_output"}


def test_branches_do_not_share_mutable_state() -> None:
    """Section 10.3: running one branch must leave the anchor untouched for the
    next one, including the message list it hands over."""
    budget = _budget()
    registry = ToolRegistry([FakePythonTool()])
    _environment, _model, journal, _natural = _run_natural(budget=budget, registry=registry)
    anchor = _anchor_from(journal)
    assert anchor.snapshot is not None
    before = anchor.snapshot.state.model_dump_json()
    messages_before = [(message.role, message.content) for message in anchor.snapshot.messages]

    first_model, first = _continue(anchor, arm=Arm.MARKER, budget=budget, registry=registry)
    second_model, second = _continue(anchor, arm=Arm.FULL, budget=budget, registry=registry)

    assert anchor.snapshot.state.model_dump_json() == before
    assert [
        (message.role, message.content) for message in anchor.snapshot.messages
    ] == messages_before
    assert anchor.snapshot.state.usage.tool_calls == 1
    # Each branch sees its own observation on an identical, untouched history:
    # the marker branch replaced it, the full-feedback branch still has the
    # traceback, and neither leaked into the other's prefix.
    assert first_model.prompts[0][-1].content == MARKER_TEXT
    assert second_model.prompts[0][-1].content == anchor.observation
    assert all(
        MARKER_TEXT not in message.content for message in second_model.prompts[0]
    )
    assert first.record["state_hash"] == second.record["state_hash"]
    # Distinct trace ids: two branches are never the same logical sample.
    assert first.record["trace_id"] != second.record["trace_id"]


def test_hidden_reference_never_reaches_the_model_input() -> None:
    budget = _budget()
    registry = ToolRegistry([FakePythonTool()])
    labeled = _labeled()
    _environment, _model, journal, _natural = _run_natural(
        budget=budget, registry=registry, labeled=labeled
    )
    anchor = _anchor_from(journal)

    initial = render_initial_messages(labeled.task, budget, registry)
    assert all(REFERENCE not in message.content for message in initial)
    assert all(REFERENCE not in message.content for message in anchor.snapshot.messages)
    assert REFERENCE not in anchor.observation

    _branch_model, outcome = _continue(
        anchor, arm=Arm.PROMPTED, budget=budget, registry=registry, labeled=labeled
    )
    assert REFERENCE not in json.dumps(outcome.record["suffix_events"])
    assert "1234567" not in json.dumps(outcome.record["suffix_events"])


def test_generation_seeds_are_derived_per_turn_and_arm() -> None:
    """Section 10.5: the requested seed must reach the client per generation."""
    budget = _budget()
    registry = ToolRegistry([FakePythonTool()])
    _environment, _model, journal, _natural = _run_natural(budget=budget, registry=registry)
    anchor = _anchor_from(journal)

    for arm in (Arm.FULL, Arm.MARKER):
        branch_model, outcome = _continue(
            anchor, arm=arm, budget=budget, registry=registry, sample_index=2
        )
        sample_id = f"{anchor.anchor_id}::a{arm.value}::2"
        expected = [
            generation_seed(
                stage="C",
                task_id=anchor.task_id,
                sample_id=sample_id,
                arm=arm.value,
                turn=turn,
            )
            for turn in range(len(branch_model.configs))
        ]
        assert [config.seed for config in branch_model.configs] == expected
        assert outcome.record["generation_seeds"][: len(expected)] == expected
        assert all(seed is not None and 0 <= seed < 2**31 for seed in expected)

    # Different arms must not share a stream for the same turn.
    assert generation_seed(
        stage="C", task_id=anchor.task_id, sample_id="x", arm="F", turn=0
    ) != generation_seed(stage="C", task_id=anchor.task_id, sample_id="x", arm="M", turn=0)


def test_infrastructure_failure_is_marked_missing_not_wrong() -> None:
    """Section 7.2: an unreachable sandbox is a missing sample, not a wrong one."""
    budget = _budget(steps=4, calls=3)
    registry = ToolRegistry([FakePythonTool()])
    _environment, _model, journal, _natural = _run_natural(budget=budget, registry=registry)
    anchor = _anchor_from(journal)
    assert anchor.eligible

    dead_registry = ToolRegistry([DeadSandboxTool()])
    _branch_model, outcome = _continue(
        anchor, arm=Arm.FULL, budget=budget, registry=dead_registry
    )

    assert outcome.infra_failure is True
    assert outcome.record["status"] == "missing"
    assert outcome.record["infra_reason"] == "tool_result:unavailable"
    # It is not silently scored as an incorrect answer.
    assert outcome.record["correct"] is False
    assert outcome.record["status"] != "complete"


def test_terminated_anchors_are_refused_by_the_environment() -> None:
    """A resume of a finished episode would silently restart it, so the
    environment must reject the state instead."""
    budget = _budget(steps=1, calls=1)
    labeled = _labeled()
    registry = ToolRegistry([FakePythonTool()])
    environment = OfflineMathEnv(labeled, budget, registry, trace_id="terminated")
    terminal_state = asyncio.run(
        environment.step(FinalAction(answer="1"), monotonic_ms=5)
    ).state

    with pytest.raises(ValueError, match="cannot resume a terminated state"):
        OfflineMathEnv(labeled, budget, registry, trace_id="x", state=terminal_state)


def test_state_hash_ignores_the_intervention() -> None:
    budget = _budget()
    registry = ToolRegistry([FakePythonTool()])
    _environment, _model, journal, _natural = _run_natural(budget=budget, registry=registry)
    snapshot = journal.snapshots[0]

    assert state_hash(snapshot) == _anchor_from(journal).state_hash
