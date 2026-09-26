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
        generated = self._model.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=config.temperature if config.temperature > 0 else None,
            eos_token_id=getattr(self._tokenizer, "eos_token_id", None),
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
            finish_reason="stop",
            model_id=self._model_id,
        )


def _first_sequence(value: Any) -> list[int]:
    first = value[0]
    return [int(token) for token in first]


def _token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(value)
