from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List

from image_red_teaming.clients.base import LLMClient
from image_red_teaming.models import ModelConfig


@dataclass(frozen=True)
class ModelChoice:
    key: str
    config: ModelConfig
    client: LLMClient


class ModelPool:
    def __init__(
        self,
        choices: List[ModelChoice],
        selection: str = "uniform",
        exploration: float = 1.0,
    ) -> None:
        if not choices:
            raise ValueError("ModelPool requires at least one model choice.")
        self._choices = list(choices)
        self._choice_by_key = {choice.key: choice for choice in choices}
        self._selection = (selection or "uniform").lower()
        self._exploration = float(exploration)
        self._stats: Dict[str, Dict[str, float]] = {
            choice.key: {"count": 0.0, "reward_sum": 0.0, "avg_reward": 0.0}
            for choice in choices
        }
        self._lock = threading.Lock()

    @classmethod
    def from_configs(
        cls,
        configs: List[ModelConfig],
        client_factory: Callable[[ModelConfig], LLMClient],
        selection: str = "uniform",
        exploration: float = 1.0,
    ) -> "ModelPool":
        seen = set()
        choices: List[ModelChoice] = []
        for idx, cfg in enumerate(configs):
            key = cls._build_key(cfg, idx, seen)
            seen.add(key)
            choices.append(ModelChoice(key=key, config=cfg, client=client_factory(cfg)))
        return cls(choices=choices, selection=selection, exploration=exploration)

    @staticmethod
    def _build_key(config: ModelConfig, idx: int, seen: set[str]) -> str:
        base = f"{config.client}:{config.model}"
        if base in seen:
            return f"{base}#{idx}"
        return base

    def select(self) -> ModelChoice:
        with self._lock:
            if self._selection == "uniform" or len(self._choices) == 1:
                return random.choice(self._choices)
            if self._selection == "ucb1":
                untried = [
                    choice for choice in self._choices if self._stats[choice.key]["count"] <= 0
                ]
                if untried:
                    return random.choice(untried)
                total = sum(stat["count"] for stat in self._stats.values())
                if total <= 0:
                    return random.choice(self._choices)
                best_choice = None
                best_score = None
                for choice in self._choices:
                    stat = self._stats[choice.key]
                    avg = stat["avg_reward"]
                    bonus = self._exploration * math.sqrt(2.0 * math.log(total) / stat["count"])
                    score = avg + bonus
                    if best_choice is None or score > float(best_score):
                        best_choice = choice
                        best_score = score
                return best_choice or random.choice(self._choices)
        raise ValueError(f"Unsupported selection policy: {self._selection}")

    def record_reward(self, key: str, reward: float) -> None:
        if key not in self._stats:
            return
        value = float(reward)
        with self._lock:
            stat = self._stats[key]
            stat["count"] += 1.0
            stat["reward_sum"] += value
            stat["avg_reward"] = stat["reward_sum"] / max(1.0, stat["count"])

    def stats_snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {key: dict(stat) for key, stat in self._stats.items()}
