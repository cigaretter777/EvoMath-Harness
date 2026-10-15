"""Single-policy agent loop that records model turns before each transition."""

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from adaptive_math.agent.environment import OfflineMathEnv, ProductMathEnv
from adaptive_math.agent.model_client import ChatMessage, GenerationConfig, ModelClient
from adaptive_math.agent.parser import parse_action
from adaptive_math.agent.prompts import PROMPT_VERSION, render_initial_messages, render_observation
from adaptive_math.agent.state import AgentState, TerminationReason
from adaptive_math.agent.trace import Trajectory
from adaptive_math.harness.spec import HarnessSpec

RUNTIME_VERSION = "runtime-v1"


@dataclass(frozen=True)
class StepSnapshot:
    """Everything needed to re-enter an episode right after one step.

    ``messages`` is the exact model-visible conversation through this step's
    observation; ``state`` is the environment state after the same step. The
    pair is what a prefix resume consumes, and it is captured live rather
    than reconstructed from events (the events do not store observations)."""

    step_index: int
    messages: tuple[ChatMessage, ...]
    state: AgentState
    terminated: bool


@dataclass
class ConversationJournal:
    """Optional per-step recorder filled by :meth:`AgentLoop.run_and_capture`."""

    snapshots: list[StepSnapshot] = field(default_factory=list)

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self.snapshots[-1].messages) if self.snapshots else []


class AgentLoop:
    """Runs one episode; when built with a HarnessSpec, stamps the trajectory
    with the spec's runtime tag and content hash, and refuses to run with
    runtime parameters that contradict the spec (no hidden drift)."""

    def __init__(
        self,
        harness: HarnessSpec | None = None,
        *,
        model_version: str | None = None,
        parser_tolerance: tuple[str, ...] = (),
    ) -> None:
        self._harness = harness
        self._model_version = model_version
        self._parser_tolerance = parser_tolerance

    async def run(
        self,
        environment: ProductMathEnv | OfflineMathEnv,
        model: ModelClient,
        generation: GenerationConfig,
        cancellation: asyncio.Event | None = None,
        *,
        prefix_messages: Sequence[ChatMessage] | None = None,
        journal: ConversationJournal | None = None,
        seed_schedule: Sequence[int] = (),
    ) -> Trajectory:
        harness = self._harness
        if harness is not None:
            if generation != harness.generation:
                raise ValueError("generation config does not match the harness spec")
            if environment.state.budget != harness.budget:
                raise ValueError("environment budget does not match the harness spec")
            if environment.registry.names != harness.tools:
                raise ValueError("environment tool set does not match the harness spec")
        if prefix_messages is None:
            messages: list[ChatMessage] = list(
                render_initial_messages(
                    environment.state.task, environment.state.budget, environment.registry
                )
            )
        else:
            # Prefix resume: the caller supplies the exact conversation the
            # model already saw, and the environment carries the matching
            # counters (events, usage, remaining budget) it inherited.
            if not prefix_messages:
                raise ValueError("prefix_messages must not be empty")
            messages = list(prefix_messages)
        started = time.monotonic()
        # Event timestamps must stay non-decreasing across the seam, so the
        # resumed clock continues past the prefix's last event instead of
        # restarting at zero.
        clock_offset_ms = environment.state.events[-1].monotonic_ms if environment.state.events else 0
        turn_index = 0
        while environment.state.termination_reason is None:
            if cancellation is not None and cancellation.is_set():
                environment.abort(TerminationReason.CANCELLED)
                break
            step_generation = generation
            if seed_schedule:
                seed = seed_schedule[turn_index] if turn_index < len(seed_schedule) else None
                if seed is not None:
                    step_generation = generation.model_copy(update={"seed": seed})
            try:
                turn = await model.generate(tuple(messages), step_generation)
            except Exception as exc:  # noqa: BLE001
                # The reason has to survive in the trajectory: "model_error"
                # alone cannot be told apart from a harness fault, and the
                # recovery protocol counts unfinished samples separately.
                environment.record_model_error(
                    f"{type(exc).__name__}: {exc}",
                    clock_offset_ms + round((time.monotonic() - started) * 1000),
                )
                environment.abort(TerminationReason.MODEL_ERROR)
                break
            monotonic_ms = clock_offset_ms + round((time.monotonic() - started) * 1000)
            environment.record_model_output(
                turn.text, turn.generated_tokens, monotonic_ms, finish_reason=turn.finish_reason
            )
            messages.append(ChatMessage(role="assistant", content=turn.text))
            parsed = parse_action(turn.text, tolerate=self._parser_tolerance)
            result = await environment.step(parsed.action, monotonic_ms)
            if result.observation is not None and not result.terminated:
                messages.append(render_observation(result.observation.content))
            if journal is not None:
                journal.snapshots.append(
                    StepSnapshot(
                        step_index=turn_index,
                        messages=tuple(messages),
                        state=environment.state,
                        terminated=result.terminated,
                    )
                )
            turn_index += 1
        state = environment.state
        assert state.termination_reason is not None
        return Trajectory(
            trace_id=environment.trace_id,
            task_id=state.task.task_id,
            events=state.events,
            final_answer=state.final_answer,
            termination_reason=state.termination_reason,
            usage=state.usage,
            runtime_version=(
                harness.runtime_tag() if harness is not None else f"{RUNTIME_VERSION}+{PROMPT_VERSION}"
            ),
            harness_spec_hash=harness.spec_hash() if harness is not None else None,
            model_version=self._model_version,
        )
