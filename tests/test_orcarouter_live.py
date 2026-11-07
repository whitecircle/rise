"""Live OrcaRouter connectivity check for the Cyber Inference provider.

Skipped unless ``ORCAROUTER_API_KEY`` is set.  Exercises the provider code path
(credential seam -> live catalog -> OpenAI-compatible request), not a bare curl.
"""

from __future__ import annotations

import os

import pytest

from image_red_teaming.clients.orcarouter import EnvApiKeySource, OrcaRouterCyberClient, resolve_api_base
from image_red_teaming.clients.orcarouter_catalog import (
    CAPABILITY_TEXT,
    CAPABILITY_VLM,
    OrcaRouterModelCatalog,
)
from image_red_teaming.models import ModelConfig

pytestmark = pytest.mark.skipif(
    not os.getenv("ORCAROUTER_API_KEY"), reason="ORCAROUTER_API_KEY not set"
)

# Internal connectivity proof model for our verification key.  User-facing
# defaults stay orcarouter/auto; this only proves the wiring.
LIVE_MODEL = "deepseek/deepseek-v4.1-flash"


def test_live_catalog_filters_text_and_vlm():
    base = resolve_api_base()
    cred = EnvApiKeySource().acquire()
    catalog = OrcaRouterModelCatalog(base, api_key=cred.key).fetch(force=True)
    assert catalog.source == "live", catalog.detail
    text_ids = {m.id for m in catalog.for_capability(CAPABILITY_TEXT)}
    vlm_ids = {m.id for m in catalog.for_capability(CAPABILITY_VLM)}
    assert "orcarouter/auto" in text_ids or "orcarouter/fusion" in text_ids
    assert not text_ids & {"jina-reranker-v2", "jina/jina-rerank-v3"}
    assert vlm_ids <= text_ids
    # Every model the filter calls a VLM must declare image input; an endpoint
    # without any VLM is fine as long as the filter itself stays correct.
    assert all("image" in m.input_modalities for m in catalog.models if m.id in vlm_ids)


def test_live_chat_completion_through_provider():
    cfg = ModelConfig(model=LIVE_MODEL, client="orcarouter")
    from image_red_teaming.clients.orcarouter import build_orcarouter_client

    client = build_orcarouter_client(cfg)
    assert isinstance(client, OrcaRouterCyberClient)
    assert client.api_base == resolve_api_base()
    result = client.complete(
        [{"role": "user", "content": "Reply with the single word: ready", "name": None}],
        max_tokens=16,
        temperature=0,
    )
    assert isinstance(result["content"], str) and result["content"].strip()
    assert result["usage"]["total_tokens"] > 0
    assert os.environ["ORCAROUTER_API_KEY"] not in str(result)
