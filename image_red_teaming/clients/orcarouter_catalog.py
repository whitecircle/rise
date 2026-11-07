"""
OrcaRouter model catalog and capability filtering.

The single source of truth for the model list is ``GET {api_base}/models`` on
the configured inference origin.  The catalog is fetched once per process,
cached on disk, and bounded in size.  When discovery fails, a small verified
seed is used so a fresh installation can still start; the seed is never mixed
into a successful live result.

Capability filters are strict: a model is offered for an entry point only when
the catalog metadata proves compatibility.  Models that do not declare their
input modalities are *not* offered for image (VLM-judge) use.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx

logger = logging.getLogger(__name__)

ORCAROUTER_PROVIDER_NAME = "OrcaRouter Cyber Inference"
ORCAROUTER_DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"
ORCAROUTER_API_KEY_ENV = "ORCAROUTER_API_KEY"

# Entry-point capabilities.  Each AI input in RISE maps to one of these.
CAPABILITY_TEXT = "text"  # attacker LLM roles (prompt/strategy/scenario/example)
CAPABILITY_VLM = "vlm"  # judge: chat model that declares `image` input

# Endpoint types the RISE clients can actually speak (OpenAI chat wire format).
_TEXT_ENDPOINT_TYPES = ("openai", "anthropic", "gemini", "openai-response")
# Endpoint types that are dedicated to a non-chat task; never offer these for chat.
_NON_TEXT_ENDPOINT_TYPES = ("image-generation", "openai-video", "jina-rerank", "embeddings")

# Bounds for live discovery so a catalog response cannot exhaust memory.
MAX_CATALOG_BYTES = 4 * 1024 * 1024
MAX_CATALOG_ITEMS = 2000
CATALOG_TIMEOUT_SECONDS = 15.0
CATALOG_TTL_SECONDS = 60 * 60
CACHE_DIR_ENV = "RISE_CACHE_DIR"


@dataclass(frozen=True)
class CatalogModel:
    """Minimal model metadata: enough to filter, never more than needed."""

    id: str
    name: str
    supported_endpoint_types: tuple = ()
    input_modalities: tuple = ()  # declared modalities; () = not declared
    context_length: Optional[int] = None
    name_declared: bool = True

    def supports_chat(self) -> bool:
        types = set(self.supported_endpoint_types)
        if types & set(_NON_TEXT_ENDPOINT_TYPES) and not (types & set(_TEXT_ENDPOINT_TYPES)):
            return False
        return bool(types & set(_TEXT_ENDPOINT_TYPES))

    def supports_image_input(self) -> bool:
        # Fail closed: a chat model that does not declare `image` is not a VLM.
        return self.supports_chat() and "image" in self.input_modalities

    def label(self) -> str:
        return f"{self.name} ({self.id})" if self.name_declared and self.name != self.id else self.id


@dataclass(frozen=True)
class ModelCatalog:
    """A catalog plus where it came from: ``live``, ``cache`` or ``seed``."""

    models: tuple
    source: str  # live | cache | seed
    fetched_at: float = 0.0
    detail: str = ""

    @property
    def degraded(self) -> bool:
        return self.source != "live"

    def for_capability(self, capability: str) -> List[CatalogModel]:
        if capability == CAPABILITY_TEXT:
            return [m for m in self.models if m.supports_chat()]
        if capability == CAPABILITY_VLM:
            return [m for m in self.models if m.supports_image_input()]
        raise ValueError(f"Unknown capability: {capability}")


# Verified cold-start seed: ids confirmed against the live /models catalog
# (verification date in the integration notes).  Metadata below is preserved
# from that response; models without declared modalities are text-only here.
_SEED_MODELS: Sequence[CatalogModel] = (
    CatalogModel(
        id="orcarouter/auto",
        name="orcarouter/auto",
        supported_endpoint_types=_TEXT_ENDPOINT_TYPES,
        input_modalities=(),
        name_declared=False,
    ),
    CatalogModel(
        id="orcarouter/fusion",
        name="orcarouter/fusion",
        supported_endpoint_types=_TEXT_ENDPOINT_TYPES,
        input_modalities=(),
        context_length=1000000,
        name_declared=False,
    ),
    CatalogModel(
        id="openai/gpt-5.5",
        name="OpenAI: GPT-5.5",
        supported_endpoint_types=("openai", "openai-response"),
        input_modalities=("file", "image", "text"),
        context_length=None,  # the public catalog reports top_provider.context_length only
        name_declared=True,
    ),
    CatalogModel(
        id="anthropic/claude-opus-4.8",
        name="Anthropic: Claude Opus 4.8",
        supported_endpoint_types=("openai", "anthropic", "openai-response"),
        input_modalities=("file", "image", "text"),
        context_length=1000000,
        name_declared=True,
    ),
    CatalogModel(
        id="google/gemini-3.5-flash",
        name="Gemini 3.5 Flash",
        supported_endpoint_types=("openai", "gemini"),
        input_modalities=("text", "image", "audio", "video", "file"),
        context_length=1048576,
        name_declared=True,
    ),
    CatalogModel(
        id="deepseek/deepseek-v4-pro",
        name="DeepSeek: DeepSeek V4 Pro",
        supported_endpoint_types=("openai", "openai-response"),
        input_modalities=("text",),
        context_length=1048576,
        name_declared=True,
    ),
)


def seed_catalog(detail: str = "live discovery unavailable") -> ModelCatalog:
    return ModelCatalog(models=tuple(_SEED_MODELS), source="seed", fetched_at=time.time(), detail=detail)


def _as_str_tuple(value: Any) -> tuple:
    if not isinstance(value, list):
        return ()
    return tuple(str(v) for v in value if isinstance(v, str) and v)


def parse_catalog_payload(payload: Any) -> List[CatalogModel]:
    """Parse a ``GET /v1/models`` body into CatalogModels, dropping bad records."""
    if not isinstance(payload, dict):
        raise ValueError("model catalog response is not a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("model catalog response has no `data` list")
    models: List[CatalogModel] = []
    seen = set()
    for item in data[:MAX_CATALOG_ITEMS]:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        if model_id in seen:
            continue
        seen.add(model_id)
        architecture = item.get("architecture")
        modalities = _as_str_tuple((architecture or {}).get("input_modalities")) if isinstance(architecture, dict) else ()
        context_length = item.get("context_length")
        models.append(
            CatalogModel(
                id=model_id,
                name=str(item.get("name") or model_id),
                supported_endpoint_types=_as_str_tuple(item.get("supported_endpoint_types")),
                input_modalities=modalities,
                context_length=context_length if isinstance(context_length, int) else None,
                name_declared=bool(item.get("name")),
            )
        )
    return models


def _cache_path() -> Optional[Path]:
    root = os.getenv(CACHE_DIR_ENV)
    if root:
        base = Path(root)
    else:
        xdg = os.getenv("XDG_CACHE_HOME")
        base = Path(xdg) / "rise" if xdg else Path.home() / ".cache" / "rise"
    return base / "orcarouter_models.json"


def _load_cache(max_age: Optional[float], api_base: Optional[str] = None) -> Optional[ModelCatalog]:
    path = _cache_path()
    if not path or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("api_base") is None:
            return None
        if api_base is not None and raw.get("api_base") != api_base:
            return None  # never restore ids discovered on a different origin
        fetched_at = float(raw.get("fetched_at") or 0.0)
        if max_age is not None and time.time() - fetched_at > max_age:
            return None
        models = parse_catalog_payload(raw.get("payload"))
    except Exception:  # corrupt cache must never break a run
        logger.debug("Ignoring unreadable OrcaRouter model cache at %s", path, exc_info=True)
        return None
    if not models:
        return None
    # Revalidate on restore: the caller re-filters by capability, and restored
    # ids are only those the catalog reported for this exact origin.
    return ModelCatalog(models=tuple(models), source="cache", fetched_at=fetched_at)


def _store_cache(api_base: str, payload: Any) -> None:
    path = _cache_path()
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"api_base": api_base, "fetched_at": time.time(), "payload": payload}),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception:
        logger.debug("Could not write OrcaRouter model cache", exc_info=True)


class OrcaRouterModelCatalog:
    """Caches one live catalog per process and per API base."""

    _lock = threading.Lock()
    _memo: Dict[str, ModelCatalog] = {}

    def __init__(
        self,
        api_base: str,
        api_key: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self._http = http_client

    @staticmethod
    def reset() -> None:
        with OrcaRouterModelCatalog._lock:
            OrcaRouterModelCatalog._memo.clear()

    def fetch(self, force: bool = False) -> ModelCatalog:
        with self._lock:
            if not force and self.api_base in self._memo:
                return self._memo[self.api_base]
            catalog = self._fetch_uncached()
            self._memo[self.api_base] = catalog
            return catalog

    def _fetch_uncached(self) -> ModelCatalog:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        client = self._http or httpx.Client(timeout=CATALOG_TIMEOUT_SECONDS, trust_env=False)
        try:
            resp = client.get(f"{self.api_base}/models", headers=headers)
            resp.raise_for_status()
            body = resp.content
            if len(body) > MAX_CATALOG_BYTES:
                raise ValueError("model catalog response too large")
            payload = json.loads(body)
            models = parse_catalog_payload(payload)
            if not models:
                raise ValueError("model catalog is empty")
        except Exception as exc:  # network, auth, shape: all fall back
            detail = _describe_error(exc)
            cached = _load_cache(max_age=None, api_base=self.api_base)
            if cached is not None:
                return ModelCatalog(
                    models=cached.models, source="cache", fetched_at=cached.fetched_at,
                    detail=f"live discovery failed ({detail}); using cached catalog",
                )
            return seed_catalog(f"live discovery failed ({detail}); using verified seed")
        finally:
            if self._http is None:
                client.close()
        _store_cache(self.api_base, payload)
        return ModelCatalog(models=tuple(models), source="live", fetched_at=time.time())


def _describe_error(exc: Exception) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return f"HTTP {status} {type(exc).__name__}"
    return type(exc).__name__


def filter_model_ids(models: Iterable[CatalogModel], capability: str) -> List[str]:
    return [m.id for m in ModelCatalog(models=tuple(models), source="live").for_capability(capability)]


def stale_model_warning(catalog: ModelCatalog, capability: str, model_id: str) -> Optional[str]:
    """Return a message when a configured model id is no longer offered."""
    offered = {m.id for m in catalog.for_capability(capability)}
    if model_id in offered:
        return None
    return (
        f"Model '{model_id}' is not in the OrcaRouter {capability} list "
        f"(catalog source: {catalog.source}). Pick another model or refresh the catalog."
    )


@dataclass
class ResolvedModel:
    """A model selection after catalog validation."""

    model_id: str
    catalog: ModelCatalog
    capability: str
    warning: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def resolve_model_selection(
    requested: Optional[str],
    catalog: OrcaRouterModelCatalog,
    capability: str,
    default_model: str,
    model_override: Optional[str] = None,
) -> ResolvedModel:
    """Validate a configured model id against the live catalog for ``capability``."""
    view = catalog.fetch()
    chosen = requested or default_model
    warning = stale_model_warning(view, capability, chosen)
    notes: List[str] = []
    if warning and model_override:
        notes.append(f"{warning} Using model_override='{model_override}' for OrcaRouter.")
        chosen = model_override
        warning = stale_model_warning(view, capability, chosen)
        if warning:
            notes.append(warning)
    return ResolvedModel(model_id=chosen, catalog=view, capability=capability, warning=warning, notes=notes)
