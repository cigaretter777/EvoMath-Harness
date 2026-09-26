"""Single-policy agent loop that records model turns before each transition."""

import asyncio
import time

from adaptive_math.agent.environment import OfflineMathEnv, ProductMathEnv
from adaptive_math.agent.model_client import ChatMessage, GenerationConfig, ModelClient
from adaptive_math.agent.parser import parse_action
from adaptive_math.agent.prompts import PROMPT_VERSION, render_initial_messages, render_observation
from adaptive_math.agent.state import TerminationReason
from adaptive_math.agent.trace import Trajectory
from adaptive_math.harness.spec import HarnessSpec

RUNTIME_VERSION = "runtime-v1"


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
    ) -> Trajectory:
        harness = self._harness
        if harness is not None:
            if generation != harness.generation:
                raise ValueError("generation config does not match the harness spec")
            if environment.state.budget != harness.budget:
                raise ValueError("environment budget does not match the harness spec")
            if environment.registry.names != harness.tools:
                raise ValueError("environment tool set does not match the harness spec")
        messages: list[ChatMessage] = list(
            render_initial_messages(environment.state.task, environment.state.budget, environment.registry)
        )
        started = time.monotonic()
        while environment.state.termination_reason is None:
            if cancellation is not None and cancellation.is_set():
                environment.abort(TerminationReason.CANCELLED)
                break
            try:
                turn = await model.generate(tuple(messages), generation)
            except Exception:  # noqa: BLE001
                environment.abort(TerminationReason.MODEL_ERROR)
                break
            monotonic_ms = round((time.monotonic() - started) * 1000)
            environment.record_model_output(turn.text, turn.generated_tokens, monotonic_ms)
            messages.append(ChatMessage(role="assistant", content=turn.text))
            parsed = parse_action(turn.text, tolerate=self._parser_tolerance)
            result = await environment.step(parsed.action, monotonic_ms)
            if result.observation is not None and not result.terminated:
                messages.append(render_observation(result.observation.content))
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
