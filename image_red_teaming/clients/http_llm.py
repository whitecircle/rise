from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from image_red_teaming.clients.base import ChatMessage, CompletionResult, LLMClient
from image_red_teaming.models import ModelConfig


class HttpLLMClient(LLMClient):
    """
    HTTPX client for a TensorZero-style /inference endpoint.

    Set `base_url` on the model config to point at your own gateway.
    """

    name = "httpx-llm"

    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None) -> None:
        super().__init__()
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.api_key = os.getenv(config.api_key_env, "")
        if not config.base_url:
            raise RuntimeError(
                "base_url is required for client: httpx. Set it on the model config "
                "to the URL of your inference gateway."
            )
        self.base_url = config.base_url
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.config.extra_headers or {})
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=60)

    @staticmethod
    def _extract_content(payload: Dict[str, Any], remove_thinking: bool = True) -> str:
        """
        TensorZero responses expose content as either:
        - {"content": [{"text": "..."}]}
        - {"output": {"content": [{"text": "..."}]}}
        """
        blocks = payload.get("content") or payload.get("output", {}).get("content")
        text = ""
        if blocks:
            text = (blocks[0] or {}).get("text", "").strip()
        elif payload.get("choices"):
            choice = (payload.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = str(message.get("content") or "").strip()
        elif isinstance(payload.get("output_text"), str):
            text = payload.get("output_text", "").strip()
        elif isinstance(payload.get("text"), str):
            text = payload.get("text", "").strip()
        if not text:
            raise ValueError("No content found in the response.")
        if remove_thinking:
            text = text.split("</think>")[-1].strip()
        return text

    def _normalize_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                content = "\n".join(str(part) for part in content)
            if not isinstance(content, str):
                content = str(content)
            normalized.append({"role": m["role"], "content": content})
        return normalized

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    def complete(
        self,
        messages: List[ChatMessage],
        remove_thinking: bool = False,
        **kwargs: Any,
    ) -> CompletionResult:
        payload_messages = self._normalize_messages(messages)
        payload: Dict[str, Any] = {
            "input": {"messages": payload_messages},
            "model_name": self.config.model,
            "params": {
            "chat_completion": {
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                }
            }
        }

        resp = self._client.post("inference", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = self._extract_content(data, remove_thinking)
        usage_totals = self._track_usage(
            data.get("usage") or data.get("output", {}).get("usage") or {}
        )
        return CompletionResult(
            content=content,
            model=data.get("model") or self.config.model,
            usage=usage_totals,
            raw=data,
        )
