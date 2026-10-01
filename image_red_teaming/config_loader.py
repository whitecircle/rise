from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict

from image_red_teaming.models import (
    CategoryExamples,
    MetaPromptTemplates,
    RunConfig,
    TechniqueEntry,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_categories(path: Path) -> list[CategoryExamples]:
    data = load_yaml(path)
    categories_raw = data.get("categories", [])
    return [CategoryExamples.model_validate(cat) for cat in categories_raw]


def load_criteria(path: Path) -> list[str]:
    data = load_yaml(path)
    if isinstance(data, list):
        criteria_raw = data
    else:
        criteria_raw = data.get("criteria", [])
    criteria = []
    for item in criteria_raw:
        if not item:
            continue
        text = str(item).strip()
        if text:
            criteria.append(text)
    return criteria


def load_meta_prompts(path: Path) -> MetaPromptTemplates:
    data = load_yaml(path)
    return MetaPromptTemplates.model_validate(data)


def load_run_config(path: Path) -> RunConfig:
    data = load_yaml(path)
    return RunConfig.model_validate(data)


def load_technique_library(path: Path) -> list[TechniqueEntry]:
    data = load_yaml(path)
    if isinstance(data, dict):
        items = data.get("techniques") or data.get("methods") or data.get("library") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []

    entries: list[TechniqueEntry] = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            entries.append(
                TechniqueEntry(name=f"Method {idx + 1}", description=item.strip())
            )
            continue
        if isinstance(item, dict):
            payload = dict(item)
            if "description" not in payload:
                for alt_key in ("text", "method", "instructions"):
                    if alt_key in payload:
                        payload["description"] = payload.pop(alt_key)
                        break
            if "name" not in payload:
                payload["name"] = f"Method {idx + 1}"
            entries.append(TechniqueEntry.model_validate(payload))
    return entries
