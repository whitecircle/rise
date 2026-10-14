from __future__ import annotations

import datetime
import json
import logging
from typing import List, Dict, Any

from pydantic import BaseModel

from image_red_teaming.clients.base import ChatMessage, LLMClient
from image_red_teaming.llm_sampling import ModelChoice, ModelPool
from image_red_teaming.logging_utils import append_jsonl, append_text
from image_red_teaming.models import CategoryExamples, ExampleItem, LoggingConfig, ModelConfig

logger = logging.getLogger(__name__)


class GeneratedExamples(BaseModel):
    """
    Pydantic model for enforcing JSON schema in LLM outputs.
    """

    examples: List[ExampleItem]


def build_example_prompt(
    category: CategoryExamples, desired: int, user_prefix: str
) -> List[ChatMessage]:
    seed_snippets = "\n".join(
        f"- text: {ex.text}\n  criteria: {ex.criteria}" for ex in category.examples
    )
    user = (
        f"{user_prefix}\n\n"
        f"Category: {category.name}\n"
        f"Seed examples:\n{seed_snippets}\n"
        f"Generate {desired} new examples with fields text and criteria."
    )
    return [
        ChatMessage(role="user", content=user, name=None),
    ]


def generate_examples_for_category(
    llm: LLMClient | None,
    category: CategoryExamples,
    desired: int,
    model_config: ModelConfig | None,
    logging: LoggingConfig,
    user_prefix: str,
    model_pool: ModelPool | None = None,
) -> List[ExampleItem]:
    if model_pool:
        choice = model_pool.select()
    elif llm and model_config:
        key = f"{model_config.client}:{model_config.model}"
        choice = ModelChoice(key=key, config=model_config, client=llm)
    else:
        raise ValueError("generate_examples_for_category requires llm/model_config or model_pool.")

    messages = build_example_prompt(category, desired, user_prefix)
    parsed, completion = choice.client.generate(
        messages,
        response_model=GeneratedExamples,
        temperature=choice.config.temperature,
        max_tokens=choice.config.max_tokens,
    )

    examples: List[ExampleItem] = []
    if parsed:
        examples = parsed.examples
    else:
        # Fallback: attempt to parse manually if content is a string with JSON.
        content = completion.get("content")
        try:
            if isinstance(content, str):
                content_obj = json.loads(content)
            else:
                content_obj = content
        except Exception:
            logger.exception("Failed to parse content as JSON for examples generation", extra={"content": content})
            content_obj = None

        if isinstance(content_obj, dict) and "examples" in content_obj:
            try:
                parsed = GeneratedExamples.model_validate(content_obj)
                examples = parsed.examples
            except Exception:
                logger.exception("Validation failed for generated examples", extra={"content": content_obj})
                examples = []

    timestamp = datetime.datetime.utcnow().isoformat()
    for ex in examples:
        ex.category = ex.category or category.name
        append_jsonl(
            logging.examples_jsonl,
            {
                "timestamp": timestamp,
                "category": ex.category,
                "text": ex.text,
                "criteria": ex.criteria,
                "model": choice.config.model,
                "model_selection": choice.key,
                "raw_usage": completion.get("usage"),
            },
        )
        append_text(
            logging.examples_text,
            f"[{timestamp}] {ex.category}: {ex.text} | criteria: {ex.criteria}",
        )
    return examples


def generate_examples(
    llm: LLMClient | None,
    categories: List[CategoryExamples],
    desired_per_category: int,
    model_config: ModelConfig | None,
    logging: LoggingConfig,
    user_prefix: str,
    model_pool: ModelPool | None = None,
) -> Dict[str, List[ExampleItem]]:
    """
    Generate examples for each category and log results.
    """
    results: Dict[str, List[ExampleItem]] = {}
    for cat in categories:
        results[cat.name] = generate_examples_for_category(
            llm=llm,
            category=cat,
            desired=desired_per_category,
            model_config=model_config,
            logging=logging,
            user_prefix=user_prefix,
            model_pool=model_pool,
        )
    return results
