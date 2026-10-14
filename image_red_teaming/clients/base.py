from __future__ import annotations

import json
import logging
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, Type

from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Any]]
    name: Optional[str]


class UsageRecord(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResult(TypedDict, total=False):
    content: Union[str, List[Any]]
    model: str
    usage: UsageRecord
    raw: Dict[str, Any]


class LLMClient(ABC):
    name: str

    def __init__(self) -> None:
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0

    @property
    def cumulative_usage(self) -> UsageRecord:
        return UsageRecord(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._total_tokens,
        )

    def _track_usage(self, usage: Dict[str, Any] | None) -> UsageRecord:
        if not usage:
            return self.cumulative_usage
        self._prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self._completion_tokens += int(usage.get("completion_tokens") or 0)
        self._total_tokens += int(usage.get("total_tokens") or 0)
        return self.cumulative_usage

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """
        Remove <think> blocks and code fences (``` or ```json) from model output.
        Handles both <think>...</think> and models that output thinking without
        an opening <think> tag (e.g., "Thinking Process:...\n</think>").
        """
        # Standard <think>...</think> and <thinking>...</thinking> blocks
        text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Models that omit the opening tag but include closing </think> or </thinking>
        text = re.sub(r"^.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE)
        if "```json" in text:
            text = text.replace("```json", "").replace("```", "")
        return text.strip()

    @staticmethod
    def _extract_thinking(text: str) -> List[str]:
        """
        Extract <think> blocks without modifying the original text.
        Handles both <think>...</think> and models that output thinking without
        an opening <think> tag.
        """
        blocks = re.findall(r"<think(?:ing)?>(.*?)</think(?:ing)?>", text, flags=re.DOTALL | re.IGNORECASE)
        if not blocks:
            # Handle models that omit opening tag but include </think> or </thinking>
            match = re.match(r"^(.*?)</think(?:ing)?>", text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                blocks = [match.group(1)]
        return blocks

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    def generate(
        self,
        messages: List[ChatMessage],
        response_model: Type[BaseModel] | None = None,
        answer_tag: str | None = None,
        **kwargs: Any,
    ) -> tuple[BaseModel | None, CompletionResult]:
        """
        Run a completion with optional Pydantic validation appended to the last user message.

        Returns (parsed_model_or_none, completion_result).
        """
        messages_copy: List[ChatMessage] = [dict(m) for m in messages]  # shallow copy
        # Prepend system_prompt from model config if set; if a pool is set,
        # sample one persona uniformly per call to break mode collapse.
        pool = getattr(self, "system_prompt_pool", None)
        if pool and not any(m.get("role") == "system" for m in messages_copy):
            chosen = random.choice(pool)
            preview = chosen.split("\n", 1)[0][:80]
            logger.debug("persona-pool sampled (%d chars): %s", len(chosen), preview)
            messages_copy.insert(0, {"role": "system", "content": chosen, "name": None})
        else:
            system_prompt = getattr(self, "system_prompt", None)
            if system_prompt and not any(m.get("role") == "system" for m in messages_copy):
                messages_copy.insert(0, {"role": "system", "content": system_prompt, "name": None})
        if response_model:
            schema_text = json.dumps(response_model.model_json_schema(), indent=2)
            for m in reversed(messages_copy):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    m["content"] = f"{m['content']}\n\nReturn JSON matching schema:\n{schema_text}"
                    break

        completion = self.complete(messages_copy, **kwargs)
        raw_content = completion.get("content")
        if isinstance(raw_content, str):
            raw_text = raw_content
        elif isinstance(raw_content, list):
            raw_text = "\n".join([str(p) for p in raw_content])
        else:
            raw_text = str(raw_content)

        think_blocks = self._extract_thinking(raw_text)
        sanitized_text = self._sanitize_text(raw_text)

        if answer_tag and f"</{answer_tag}>" not in sanitized_text:
            logger.warning("Answer tag '%s' not found in response: %s, raw content: %s", answer_tag, sanitized_text, raw_content)
            raise ValueError(f"Answer tag '{answer_tag}' not found in response")

        parsed: BaseModel | None = None
        if response_model:
            try:
                parsed = response_model.model_validate_json(sanitized_text)
            except (ValidationError, ValueError) as exc:
                logger.warning("Pydantic validation failed for %s: %s", response_model.__name__, exc)


        completion_with_sanitized: CompletionResult = dict(completion)
        completion_with_sanitized["content"] = sanitized_text
        completion_with_sanitized["raw_content"] = raw_text
        if think_blocks:
            completion_with_sanitized["think_blocks"] = think_blocks
        return parsed, completion_with_sanitized

    @abstractmethod
    def complete(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> CompletionResult:
        """
        Execute a chat/completions request using provider-specific client.
        """
        raise NotImplementedError
