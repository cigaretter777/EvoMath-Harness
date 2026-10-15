import asyncio

import pytest

from adaptive_math.agent.model_client import ChatMessage, GenerationConfig
from adaptive_math.agent.transformers_client import TransformersModelClient


class FakeTokenizer:
    eos_token_id = 0
    chat_template = "fake-template"

    def __init__(self, completion: list[int] | None = None) -> None:
        self.messages: list[dict[str, str]] | None = None
        self.completion = completion if completion is not None else [20, 21]

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
        self.messages = messages
        return [10, 11]

    def decode(self, tokens: list[int], **kwargs: object) -> str:
        assert tokens == self.completion
        return '<final>{"answer":"2"}</final><|end|>'


class FakeModel:
    def __init__(self, completion: list[int] | None = None) -> None:
        self.kwargs: dict[str, object] | None = None
        self.completion = completion if completion is not None else [20, 21]

    def generate(self, input_ids: list[int], **kwargs: object) -> list[list[int]]:
        self.kwargs = kwargs
        assert input_ids == [10, 11]
        return [[10, 11, *self.completion]]


def test_transformers_client_uses_chat_template_and_stops_on_configured_marker() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    client = TransformersModelClient(tokenizer, model, model_id="fake", stop_strings=("<|end|>",))

    turn = asyncio.run(
        client.generate(
            (ChatMessage(role="system", content="rules"), ChatMessage(role="user", content="problem")),
            GenerationConfig(max_new_tokens=8),
        )
    )

    assert tokenizer.messages == [{"role": "system", "content": "rules"}, {"role": "user", "content": "problem"}]
    assert turn.text == '<final>{"answer":"2"}</final>'
    assert turn.prompt_tokens == 2 and turn.generated_tokens == 2
    assert model.kwargs is not None and model.kwargs["max_new_tokens"] == 8
    # No seed requested -> no generator override, and the run ended early
    # rather than at the ceiling.
    assert "generator" not in model.kwargs
    assert turn.finish_reason == "stop"


def test_transformers_client_threads_the_requested_seed_into_sampling() -> None:
    """A declared seed must really reach sampling, not just the JSON.

    HF's ``generate`` has no per-call RNG argument (passing ``generator``
    raises "not used by the model" on transformers 4.57), so the client seeds
    the global RNG under ``fork_rng``. The double records the seed it sees
    inside the call, and the test checks the enclosing state comes back
    unchanged. The torch runtime is optional for the CPU-only venv, so skip
    there."""
    torch = pytest.importorskip("torch")

    class SeedObservingModel(FakeModel):
        observed_seed: int | None = None

        def generate(self, input_ids: list[int], **kwargs: object) -> list[list[int]]:
            self.observed_seed = int(torch.initial_seed())
            return super().generate(input_ids, **kwargs)

    tokenizer = FakeTokenizer()
    model = SeedObservingModel()
    client = TransformersModelClient(tokenizer, model, model_id="fake")
    before = int(torch.initial_seed())

    asyncio.run(
        client.generate(
            (ChatMessage(role="user", content="problem"),),
            GenerationConfig(max_new_tokens=8, temperature=0.7, seed=7),
        )
    )

    assert model.observed_seed == 7
    # fork_rng restored the surrounding state, so one request cannot shift the
    # randomness of the next one.
    assert int(torch.initial_seed()) == before

    asyncio.run(
        client.generate(
            (ChatMessage(role="user", content="problem"),),
            GenerationConfig(max_new_tokens=8, temperature=0.7, seed=99),
        )
    )
    assert model.observed_seed == 99


def test_finish_reason_reports_length_when_the_ceiling_is_hit() -> None:
    """The client used to hardcode "stop", so truncation was invisible."""
    completion = [20, 21, 22, 23]  # 4 tokens, no EOS, ceiling is 4
    tokenizer = FakeTokenizer(completion=completion)
    model = FakeModel(completion=completion)
    client = TransformersModelClient(tokenizer, model, model_id="fake")

    turn = asyncio.run(
        client.generate(
            (ChatMessage(role="user", content="problem"),),
            GenerationConfig(max_new_tokens=4),
        )
    )

    assert turn.finish_reason == "length"
    assert turn.generated_tokens == 4


def test_finish_reason_stays_stop_when_eos_lands_exactly_on_the_ceiling() -> None:
    completion = [20, 21, 0]  # ends on the tokenizer's EOS within the ceiling
    tokenizer = FakeTokenizer(completion=completion)
    model = FakeModel(completion=completion)
    client = TransformersModelClient(tokenizer, model, model_id="fake")

    turn = asyncio.run(
        client.generate(
            (ChatMessage(role="user", content="problem"),),
            GenerationConfig(max_new_tokens=3),
        )
    )

    assert turn.finish_reason == "stop"
