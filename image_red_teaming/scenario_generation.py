from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from image_red_teaming.clients.base import ChatMessage, LLMClient
from image_red_teaming.llm_sampling import ModelChoice, ModelPool
from image_red_teaming.logging_utils import append_jsonl
from image_red_teaming.models import (
    ExampleItem,
    LoggingConfig,
    ModelConfig,
    ScenarioGenerationConfig,
)

logger = logging.getLogger(__name__)


class ScenarioItem(BaseModel):
    """
    Single scenario stub entry.
    """

    text: str


class GeneratedScenarios(BaseModel):
    """
    Pydantic model for enforcing JSON schema in LLM outputs.
    """

    scenarios: List[ScenarioItem]


class ScenarioGenerator:
    """
    Generates diverse scenario stubs for a given criterion.
    """

    def __init__(
        self,
        llm: LLMClient | None,
        model_config: ModelConfig | None,
        user_prefix: str,
        prompt_lines: List[str],
        high_feedback_label: str,
        low_feedback_label: str,
        model_pool: ModelPool | None = None,
    ) -> None:
        self.llm = llm
        self.model_config = model_config
        self.user_prefix = user_prefix
        self.prompt_lines = prompt_lines
        self.high_feedback_label = high_feedback_label
        self.low_feedback_label = low_feedback_label
        self.model_pool = model_pool
        self.last_model_key: Optional[str] = None
        self.last_model_name: Optional[str] = None
        if not self.model_pool and (self.llm is None or self.model_config is None):
            raise ValueError("ScenarioGenerator requires llm/model_config or model_pool.")

    def _select_model(self) -> ModelChoice:
        if self.model_pool:
            return self.model_pool.select()
        if not self.llm or not self.model_config:
            raise RuntimeError("Model selection requested without llm/model_config.")
        key = f"{self.model_config.client}:{self.model_config.model}"
        return ModelChoice(key=key, config=self.model_config, client=self.llm)

    def _format_feedback(self, entries: List[Dict[str, Any]]) -> List[str]:
        if not entries:
            return []
        lines = []
        for item in entries:
            lines.append(
                " ".join(
                    [
                        f"- text={item.get('text')}",
                        f"min_score={item.get('min_score'):.3f}",
                        f"max_score={item.get('max_score'):.3f}",
                        f"n_prompts={item.get('num_prompts')}",
                    ]
                )
            )
        return lines

    def _build_messages(
        self,
        criteria: str,
        desired: int,
        high_feedback: List[Dict[str, Any]],
        low_feedback: List[Dict[str, Any]],
    ) -> List[ChatMessage]:
        high_block = "\n".join(self._format_feedback(high_feedback))
        low_block = "\n".join(self._format_feedback(low_feedback))
        format_vars = {
            "criteria": criteria,
            "desired": desired,
            "high_header": self.high_feedback_label if high_block else "",
            "high_feedback": high_block,
            "low_header": self.low_feedback_label if low_block else "",
            "low_feedback": low_block,
        }
        lines = [self.user_prefix.strip()]
        if self.prompt_lines:
            lines.append("")
            for line in self.prompt_lines:
                formatted = line.format(**format_vars)
                if formatted.strip():
                    lines.append(formatted)
        return [ChatMessage(role="user", content="\n".join(lines), name=None)]

    @staticmethod
    def _fix_json(text: str) -> str:
        """Fix common JSON issues from LLM output: trailing commas, single quotes."""
        import re
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def _parse_fallback(self, content: Any) -> List[ScenarioItem]:
        content_obj = None
        if isinstance(content, str):
            # Try standard JSON first
            try:
                content_obj = json.loads(content)
            except Exception:
                pass
            # Try after fixing common issues
            if content_obj is None:
                try:
                    content_obj = json.loads(self._fix_json(content))
                except Exception:
                    pass
            # Try extracting JSON block from response
            if content_obj is None:
                import re
                match = re.search(r'\{[^{}]*"scenarios"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
                if match:
                    try:
                        content_obj = json.loads(self._fix_json(match.group(0)))
                    except Exception:
                        pass
            if content_obj is None:
                logger.warning(
                    "Failed to parse content as JSON for scenario generation (len=%d)", len(content) if content else 0
                )
                return []
        else:
            content_obj = content

        if isinstance(content_obj, dict) and "scenarios" in content_obj:
            try:
                parsed = GeneratedScenarios.model_validate(content_obj)
                return parsed.scenarios
            except Exception:
                # Try extracting text fields manually
                scenarios = content_obj.get("scenarios", [])
                results = []
                for item in scenarios:
                    if isinstance(item, dict) and "text" in item:
                        results.append(ScenarioItem(text=str(item["text"])))
                if results:
                    return results
                logger.warning(
                    "Validation failed for generated scenarios", extra={"content": content_obj}
                )
        return []

    def generate(
        self,
        criteria: str,
        desired: int,
        high_feedback: Optional[List[Dict[str, Any]]] = None,
        low_feedback: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ScenarioItem]:
        choice = self._select_model()
        self.last_model_key = choice.key
        self.last_model_name = choice.config.model
        messages = self._build_messages(
            criteria,
            desired,
            high_feedback or [],
            low_feedback or [],
        )
        parsed, completion = choice.client.generate(
            messages,
            response_model=GeneratedScenarios,
            temperature=choice.config.temperature,
            max_tokens=choice.config.max_tokens,
        )

        if parsed:
            return parsed.scenarios
        content = completion.get("content")
        return self._parse_fallback(content)


class ScenarioManager:
    """
    Maintains scenario stubs, updates scenario stats, and refreshes stubs with feedback.
    """

    def __init__(
        self,
        criteria: List[str],
        generator: ScenarioGenerator,
        config: ScenarioGenerationConfig,
        logging_config: LoggingConfig,
        model_config: ModelConfig | None,
    ) -> None:
        self.criteria = criteria
        self.generator = generator
        self.config = config
        self.logging = logging_config
        self.model_config = model_config
        self._memory = self._load_memory()
        self._scenarios: Dict[str, List[str]] = {}

    def _load_memory(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        path = Path(self.logging.scenario_memory_json)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            logger.exception("Failed to load scenario memory", extra={"path": str(path)})
        return {}

    def _save_memory(self) -> None:
        path = Path(self.logging.scenario_memory_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._memory, f, ensure_ascii=True, indent=2)

    def _extract_score(self, record: Dict[str, Any]) -> Optional[float]:
        max_score = record.get("max_score")
        if isinstance(max_score, (int, float)):
            return float(max_score)
        avg_score = record.get("avg_score")
        if isinstance(avg_score, (int, float)):
            return float(avg_score)
        score = record.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        scores = record.get("image_scores")
        if isinstance(scores, list):
            numeric = [float(val) for val in scores if isinstance(val, (int, float))]
            if numeric:
                return max(numeric)
        return None

    def update_from_rollouts(self, rollouts: List[Dict[str, Any]]) -> None:
        for rec in rollouts:
            ex = rec.get("example", {})
            criteria = ex.get("criteria")
            text = ex.get("text")
            if not criteria or not text:
                continue
            score = self._extract_score(rec)
            if score is None:
                continue
            by_criteria = self._memory.setdefault(criteria, {})
            stats = by_criteria.setdefault(
                text,
                {"count": 0, "sum_score": 0.0, "min_score": score, "max_score": score},
            )
            stats["count"] = int(stats.get("count", 0)) + 1
            stats["sum_score"] = float(stats.get("sum_score", 0.0)) + float(score)
            stats["min_score"] = min(float(stats.get("min_score", score)), float(score))
            stats["max_score"] = max(float(stats.get("max_score", score)), float(score))

    def _stats_for_criteria(self, criteria: str) -> List[Dict[str, Any]]:
        by_text = self._memory.get(criteria) or {}
        stats = []
        for text, entry in by_text.items():
            count = int(entry.get("count", 0))
            if count <= 0:
                continue
            sum_score = float(entry.get("sum_score", 0.0))
            stats.append(
                {
                    "text": text,
                    "avg_score": sum_score / count,
                    "min_score": float(entry.get("min_score", 0.0)),
                    "max_score": float(entry.get("max_score", 0.0)),
                    "num_prompts": count,
                }
            )
        return stats

    def _split_feedback(
        self, stats: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        max_count = max(0, int(self.config.feedback_examples))
        if max_count <= 0 or not stats:
            return [], []
        sorted_high = sorted(stats, key=lambda item: item.get("max_score", 0.0))
        sorted_low = sorted(stats, key=lambda item: item.get("max_score", 0.0))
        if len(stats) <= max_count:
            if len(sorted_low) <= 1:
                return sorted_high, []
            low_count = max(1, len(sorted_low) // 2)
            high_count = len(sorted_low) - low_count
        else:
            low_count = max(1, max_count // 2)
            high_count = max_count - low_count
            if high_count <= 0:
                high_count = 1
                low_count = max_count - high_count
        high = sorted_high[-high_count:]
        high_ids = {id(item) for item in high}
        low = [item for item in sorted_low[:low_count] if id(item) not in high_ids]
        return high, low

    def _dedupe(self, scenarios: List[ScenarioItem]) -> List[str]:
        seen = set()
        output = []
        for item in scenarios:
            text = item.text.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            output.append(text)
        return output

    def _log_scenarios(
        self,
        criteria: str,
        scenarios: List[str],
        model_name: Optional[str] = None,
    ) -> None:
        timestamp = datetime.datetime.utcnow().isoformat()
        fallback_model = self.model_config.model if self.model_config else "unknown"
        model_label = model_name or fallback_model
        for text in scenarios:
            append_jsonl(
                self.logging.scenario_examples_jsonl,
                {
                    "timestamp": timestamp,
                    "criteria": criteria,
                    "text": text,
                    "model": model_label,
                },
            )

    def _generate_for_criteria(self, criteria: str, fallback: List[str]) -> List[str]:
        if not self.config.enabled:
            return [criteria]
        stats = self._stats_for_criteria(criteria)
        high, low = self._split_feedback(stats)
        scenarios = self._dedupe(
            self.generator.generate(
                criteria=criteria,
                desired=int(self.config.scenarios_per_criteria),
                high_feedback=high,
                low_feedback=low,
            )
        )
        if scenarios:
            return scenarios
        return fallback or [criteria]

    def refresh(self) -> None:
        previous = self._scenarios
        self._scenarios = {}
        for criteria in self.criteria:
            scenarios = self._generate_for_criteria(criteria, previous.get(criteria, []))
            desired = max(1, int(self.config.scenarios_per_criteria))
            if len(scenarios) < desired:
                for item in previous.get(criteria, []):
                    if len(scenarios) >= desired:
                        break
                    if item not in scenarios:
                        scenarios.append(item)
            self._scenarios[criteria] = scenarios
            model_label = self.generator.last_model_key or self.generator.last_model_name
            self._log_scenarios(criteria, scenarios, model_name=model_label)
        self._save_memory()

    def maybe_refresh(self, iteration: int) -> bool:
        interval = int(self.config.refresh_interval)
        if interval <= 0 or iteration % interval != 0:
            return False
        self.refresh()
        return True

    def current_examples(self) -> List[ExampleItem]:
        examples = []
        for criteria, scenarios in self._scenarios.items():
            for text in scenarios:
                examples.append(ExampleItem(text=text, criteria=criteria))
        return examples
