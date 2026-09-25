"""Backend-neutral vectorized core for a future pinned verl-agent adapter."""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy import typing as npt

from adaptive_math.agent.actions import ToolAction
from adaptive_math.agent.environment import OfflineMathEnv
from adaptive_math.agent.model_client import ChatMessage
from adaptive_math.agent.parser import TOLERANCE_RULES, parse_action
from adaptive_math.agent.prompts import render_initial_messages, render_observation
from adaptive_math.agent.trace import Trajectory
from adaptive_math.core.types import Budget, JSONValue, LabeledMathTask
from adaptive_math.reward import RewardConfig
from adaptive_math.tools.registry import ToolRegistry
from adaptive_math.training.reward_bridge import reward_for_trajectory

if TYPE_CHECKING:
    class _EnvironmentManagerBase:
        def __init__(self, envs: object, projection_f: object, config: object) -> None:
            self.envs = envs
            self.projection_f = projection_f
            self.config = config
else:
    try:  # Imported only in the cloud image where the pinned backend is installed.
        from agent_system.environments.base import EnvironmentManagerBase as _EnvironmentManagerBase
    except ImportError:  # pragma: no cover - exercised by the cloud backend contract.
        class _EnvironmentManagerBase:
            def __init__(self, envs: object, projection_f: object, config: object) -> None:
                self.envs = envs
                self.projection_f = projection_f
                self.config = config


@dataclass(frozen=True)
class RolloutTransition:
    env_id: str
    group_id: str
    policy_version: str
    observation: str | None
    reward: float
    done: bool
    info: dict[str, JSONValue]


class MathRolloutManager:
    """Owns isolated OfflineMathEnv instances for synchronous policy groups.

    The upstream collector is fixed-width: it generates one action per batch
    row on every step and requires one reward/done/observation per row in
    return.  Terminated environments therefore stay in the batch as no-op
    rows that repeat their terminal info and last observation.
    """

    def __init__(
        self,
        budget: Budget,
        registry: ToolRegistry,
        *,
        reward_config: RewardConfig | None = None,
        max_generated_tokens: int = 0,
    ) -> None:
        if max_generated_tokens < 0:
            raise ValueError("max_generated_tokens must be non-negative")
        self._budget = budget
        self._registry = registry
        self._reward_config = reward_config
        self._max_generated_tokens = max_generated_tokens
        self._environments: dict[str, OfflineMathEnv] = {}
        self._groups: dict[str, str] = {}
        self._conversations: dict[str, list[ChatMessage]] = {}
        self._last_observations: dict[str, str | None] = {}
        self._terminal_infos: dict[str, dict[str, JSONValue]] = {}
        self._policy_version = ""
        self._step_index = 0

    def reset(
        self, tasks: list[LabeledMathTask], *, group_size: int, policy_version: str
    ) -> list[str]:
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if not policy_version:
            raise ValueError("policy_version is required")
        self._environments = {}
        self._groups = {}
        self._conversations = {}
        self._last_observations = {}
        self._terminal_infos = {}
        self._policy_version = policy_version
        self._step_index = 0
        observations: list[str] = []
        for task in tasks:
            for sample_index in range(group_size):
                base = f"{task.task.task_id}:{sample_index}"
                env_id = base
                # The upstream collector repeats rows of the same task within
                # one batch (GRPO groups), so a bare task:sample id can collide.
                suffix = 2
                while env_id in self._environments:
                    env_id = f"{base}:{suffix}"
                    suffix += 1
                environment = OfflineMathEnv(
                    task, self._budget, self._registry, trace_id=f"rollout:{env_id}"
                )
                self._environments[env_id] = environment
                self._groups[env_id] = task.task.task_id
                self._conversations[env_id] = list(
                    render_initial_messages(environment.state.task, self._budget, self._registry)
                )
                observations.append(self._render_conversation(env_id))
        return observations

    async def step(self, text_actions: list[str]) -> list[RolloutTransition]:
        if len(text_actions) != len(self._environments):
            raise ValueError("text_actions must have one entry for each environment")
        self._step_index += 1

        async def one(env_id: str, environment: OfflineMathEnv, text: str) -> RolloutTransition:
            if environment.state.termination_reason is not None:
                # Fixed-width contract: terminated rows accept a no-op action
                # and keep repeating their terminal info and last observation.
                return RolloutTransition(
                    env_id=env_id,
                    group_id=self._groups[env_id],
                    policy_version=self._policy_version,
                    observation=self._last_observations[env_id],
                    reward=0.0,
                    done=True,
                    info=self._terminal_infos[env_id],
                )
            environment.record_model_output(text, generated_tokens=0, monotonic_ms=self._step_index)
            self._conversations[env_id].append(ChatMessage(role="assistant", content=text))
            parsed = parse_action(text, tolerate=TOLERANCE_RULES)
            result = await environment.step(parsed.action, self._step_index)
            evaluation = environment.evaluate() if result.terminated else None
            reward = evaluation.reward if evaluation is not None else 0.0
            info: dict[str, JSONValue] = {
                "trace_id": environment.trace_id,
                "policy_version": self._policy_version,
                "termination_reason": result.state.termination_reason.value
                if result.state.termination_reason is not None
                else None,
                "verifier_status": evaluation.status.value if evaluation is not None else None,
                "won": float(reward > 0.0),
                "is_action_valid": parsed.error is None,
                "tool_calling": 1.0 if isinstance(parsed.action, ToolAction) else 0.0,
            }
            if evaluation is not None and self._reward_config is not None:
                state = environment.state
                assert state.termination_reason is not None
                breakdown = reward_for_trajectory(
                    Trajectory(
                        trace_id=environment.trace_id,
                        task_id=state.task.task_id,
                        events=state.events,
                        final_answer=state.final_answer,
                        termination_reason=state.termination_reason,
                        usage=state.usage,
                        runtime_version="verl-agent-adapter-v1",
                    ),
                    evaluation,
                    self._budget,
                    max_generated_tokens=self._max_generated_tokens,
                    config=self._reward_config,
                )
                reward = breakdown.total
                info["reward_version"] = breakdown.reward_version
                info["reward_components"] = cast(JSONValue, breakdown.components)
            observation = result.observation.content if result.observation is not None else None
            if observation is not None:
                self._conversations[env_id].append(render_observation(observation))
            self._last_observations[env_id] = observation
            if result.terminated:
                self._terminal_infos[env_id] = dict(info)
            return RolloutTransition(
                env_id=env_id,
                group_id=self._groups[env_id],
                policy_version=self._policy_version,
                observation=observation,
                reward=reward,
                done=result.terminated,
                info=info,
            )

        return list(await asyncio.gather(*(one(env_id, environment, text) for (env_id, environment), text in zip(self._environments.items(), text_actions, strict=True))))

    def build_text_obs(self) -> list[str]:
        """Full per-environment turn context, one entry per batch row.

        Terminated environments repeat their final context so the upstream
        fixed-width loop can keep indexing every row.
        """
        return [self._render_conversation(env_id) for env_id in self._environments]

    def _render_conversation(self, env_id: str) -> str:
        blocks: list[str] = []
        for message in self._conversations[env_id]:
            if message.role == "assistant":
                blocks.append(f"Assistant:\n{message.content}")
            elif message.role == "tool":
                blocks.append(f"Observation:\n{message.content}")
            else:
                blocks.append(message.content)
        return "\n\n".join(blocks)


class VerlMathEnvironmentManager(_EnvironmentManagerBase):
    """Pinned verl-agent environment adapter backed only by production runtime code.

    The class is importable in the lightweight local environment.  In the
    cloud image it subclasses the exact ``EnvironmentManagerBase`` supplied by
    the SHA recorded in ``third_party/manifest.json``.
    """

    def __init__(
        self,
        tasks: list[LabeledMathTask],
        budget: Budget,
        registry: ToolRegistry,
        config: object,
        *,
        policy_version: str,
        task_lookup: dict[str, LabeledMathTask] | None = None,
        reward_config: RewardConfig | None = None,
        max_generated_tokens: int = 0,
    ) -> None:
        try:
            group_size = int(config.env.rollout.n)  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("config.env.rollout.n must be a positive group_size") from exc
        if group_size <= 0:
            raise ValueError("config.env.rollout.n must be a positive group_size")
        if not tasks and not task_lookup:
            raise ValueError("at least one labeled task is required")
        super().__init__(None, lambda actions: (actions, [True] * len(actions)), config)
        self._tasks = tasks
        self._task_lookup = task_lookup
        self._group_size = group_size
        self._policy_version = policy_version
        self._manager = MathRolloutManager(
            budget,
            registry,
            reward_config=reward_config,
            max_generated_tokens=max_generated_tokens,
        )

    def reset(self, kwargs: object) -> tuple[dict[str, object], list[dict[str, object]]]:
        tasks = self._tasks
        if self._task_lookup is not None:
            from adaptive_math.training.verl_agent_adapter import task_ids_from_reset_kwargs

            task_ids = task_ids_from_reset_kwargs(kwargs)
            try:
                tasks = [self._task_lookup[task_id] for task_id in task_ids]
            except KeyError as exc:
                raise ValueError(f"unknown adaptive-math task id {exc.args[0]!r}") from exc
        # One environment per gen-batch row: GRPO groups are the upstream
        # collector's uid blocks, never an environment-side expansion.
        text = self._manager.reset(tasks, group_size=1, policy_version=self._policy_version)
        return {"text": text, "image": None, "anchor": text.copy()}, [{} for _ in text]

    def step(
        self, text_actions: list[str]
    ) -> tuple[
        dict[str, object],
        npt.NDArray[np.float32],
        npt.NDArray[np.bool_],
        list[dict[str, JSONValue]],
    ]:
        transitions = asyncio.run(self._manager.step(text_actions))
        infos: list[dict[str, JSONValue]] = []
        for transition in transitions:
            info = dict(transition.info)
            info.setdefault("won", float(transition.reward > 0.0))
            info.setdefault("is_action_valid", True)
            infos.append(info)
        return (
            {"text": self._manager.build_text_obs(), "image": None, "anchor": None},
            np.asarray([transition.reward for transition in transitions], dtype=np.float32),
            np.asarray([transition.done for transition in transitions], dtype=bool),
            infos,
        )

    def success_evaluator(
        self, *args: object, **kwargs: object
    ) -> dict[str, npt.NDArray[np.float32]]:
        del args
        total_batch_list = kwargs.get("total_batch_list")
        total_infos = kwargs.get("total_infos")
        if not isinstance(total_batch_list, list) or not isinstance(total_infos, list):
            raise TypeError("total_batch_list and total_infos are required")
        successes: list[float] = []
        for batches, infos in zip(total_batch_list, total_infos, strict=True):
            if not isinstance(batches, list) or not isinstance(infos, list):
                raise TypeError("total batch entries must be lists")
            for batch, info in zip(reversed(batches), reversed(infos), strict=True):
                if isinstance(batch, dict) and batch.get("active_masks") and isinstance(info, dict):
                    won = info.get("won")
                    if not isinstance(won, (float, int)):
                        raise ValueError("terminal rollout info lacks won")
                    successes.append(float(won))
                    break
            else:
                raise ValueError("each batch requires an active terminal step")
        return {"success_rate": np.asarray(successes, dtype=np.float32)}
