from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from image_red_teaming.clients.base import ChatMessage, CompletionResult, LLMClient


class OpenRouterClient(LLMClient):
    """
    OpenRouter LLM/VLM client using the OpenAI-compatible chat completions API.

    Supports multi-modal prompts by accepting `content` as strings or a list
    of text/image parts following the OpenAI schema.
    """

    name = "openrouter"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str | None = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Create an API key at https://openrouter.ai/keys and export it."
            )

        self.model = model
        self.client = OpenAI(
            api_key=key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            default_headers=default_headers,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    def complete(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> CompletionResult:


        messages = [
            {"role": x["role"], "content": x["content"]} for x in messages
        ]
        extra_body = getattr(self, "extra_body", None) or {}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body if extra_body else None,
            **kwargs,
        )

        # Convert to plain dict to make downstream logging/serialization simple.
        if hasattr(response, "model_dump"):
            raw: Dict[str, Any] = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw = response.to_dict()  # type: ignore[assignment]
        else:
            raw = {"response": response}

        usage_dict: Dict[str, Any] = raw.get("usage") or {}
        usage_totals = self._track_usage(usage_dict)

        first_choice = response.choices[0]
        content = first_choice.message.content
        # vllm/Qwen3.5 may put thinking in reasoning field, content is just the answer
        reasoning = getattr(first_choice.message, "reasoning_content", None) or getattr(first_choice.message, "reasoning", None)
        if not reasoning:
            # Check raw response for reasoning field
            raw_msg = raw.get("choices", [{}])[0].get("message", {})
            reasoning = raw_msg.get("reasoning_content") or raw_msg.get("reasoning")
        if reasoning and isinstance(reasoning, str):
            think_block = f"<think>\n{reasoning}\n</think>\n\n"
            if content:
                content = think_block + content
            else:
                content = think_block

        return CompletionResult(
            content=content,
            model=response.model or self.model,
            usage=usage_totals,
            raw=raw,
        )
