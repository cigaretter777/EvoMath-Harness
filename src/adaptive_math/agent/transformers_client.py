"""Local Hugging Face adapter with delayed heavyweight imports."""

import asyncio
from typing import Any

from adaptive_math.agent.model_client import ChatMessage, GenerationConfig, ModelTurn


class TransformersModelClient:
    """Render messages through the tokenizer's own chat template before generation."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        model_id: str,
        stop_strings: tuple[str, ...] = (),
    ) -> None:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("tokenizer must define a chat_template")
        self._tokenizer = tokenizer
        self._model = model
        self._model_id = model_id
        self._stop_strings = stop_strings

    @classmethod
    def from_pretrained(
        cls, model_id: str, *, device: str = "auto", dtype: str = "auto", adapter: str | None = None
    ) -> "TransformersModelClient":
        try:
            from transformers import (  # type: ignore[import-not-found, unused-ignore]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional cloud runtime dependencies (transformers and torch) to load a model."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(model_id)  # type: ignore[no-untyped-call, unused-ignore]
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
        if adapter is not None:
            from peft import PeftModel  # type: ignore[import-not-found, unused-ignore]

            model = PeftModel.from_pretrained(model, adapter)
        return cls(tokenizer, model, model_id=model_id)

    async def generate(
        self, messages: tuple[ChatMessage, ...], config: GenerationConfig
    ) -> ModelTurn:
        return await asyncio.to_thread(self._generate, messages, config)

    def _generate(self, messages: tuple[ChatMessage, ...], config: GenerationConfig) -> ModelTurn:
        payload = [message.model_dump() for message in messages]
        input_ids = self._tokenizer.apply_chat_template(
            payload,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        get_input_embeddings = getattr(self._model, "get_input_embeddings", None)
        if hasattr(input_ids, "to") and callable(get_input_embeddings):
            input_device = get_input_embeddings().weight.device
            input_ids = input_ids.to(input_device)
        prompt_tokens = _token_count(input_ids)
        eos_ids = _eos_token_ids(getattr(self._tokenizer, "eos_token_id", None))
        generated = _call_generate(
            self._model,
            input_ids,
            seed=config.seed,
            kwargs={
                "max_new_tokens": config.max_new_tokens,
                "do_sample": config.temperature > 0,
                "temperature": config.temperature if config.temperature > 0 else None,
                "eos_token_id": getattr(self._tokenizer, "eos_token_id", None),
            },
        )
        sequence = _first_sequence(generated)
        completion = sequence[prompt_tokens:]
        text = self._tokenizer.decode(completion, skip_special_tokens=True)
        for marker in self._stop_strings:
            text = text.split(marker, 1)[0]
        return ModelTurn(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(completion),
            finish_reason=_finish_reason(completion, eos_ids, config.max_new_tokens),
            model_id=self._model_id,
        )


def _first_sequence(value: Any) -> list[int]:
    first = value[0]
    return [int(token) for token in first]


def _call_generate(model: Any, input_ids: Any, *, seed: int | None, kwargs: dict[str, Any]) -> Any:
    """Call ``model.generate``, honouring a requested seed.

    Hugging Face's ``generate`` has no per-call RNG argument -- passing
    ``generator`` raises "The following `model_kwargs` are not used by the
    model" on transformers 4.57 -- so the only way to make a declared seed real
    is to seed torch's global RNG and restore it afterwards. ``fork_rng``
    snapshots and restores both the CPU and the listed CUDA device states, so
    one call cannot leak randomness into the next.

    The callers here run one generation at a time; a concurrent caller would
    need its own lock, because the seeding window is process-global.
    """
    if seed is None:
        return model.generate(input_ids, **kwargs)
    import torch  # type: ignore[import-not-found, unused-ignore]

    device = getattr(input_ids, "device", None)
    devices = [device] if getattr(device, "type", None) == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return model.generate(input_ids, **kwargs)


def _eos_token_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(token) for token in value}


def _finish_reason(completion: list[int], eos_ids: set[int], max_new_tokens: int) -> str:
    """``length`` only when the ceiling was hit without an end-of-sequence
    token; the previous hardcoded ``stop`` could not distinguish truncation."""
    if len(completion) < max_new_tokens:
        return "stop"
    return "stop" if completion and completion[-1] in eos_ids else "length"


def _token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(value)
