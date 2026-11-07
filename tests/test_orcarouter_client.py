"""OrcaRouter Cyber Inference provider tests.  All credentials are fake."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.parse
import urllib.request

import httpx
import pytest

from image_red_teaming.clients import orcarouter as orca
from image_red_teaming.clients.orcarouter_catalog import (
    ORCAROUTER_API_KEY_ENV,
    CAPABILITY_TEXT,
    CAPABILITY_VLM,
    ModelCatalog,
    OrcaRouterModelCatalog,
    parse_catalog_payload,
    stale_model_warning,
)

from image_red_teaming.models import ModelConfig

FAKE_KEY = "sk-orca-" + "test" * 8
FAKE_VERIFIER = "v" * 43

CATALOG_BODY = {
    "data": [
        {"id": "orcarouter/auto", "supported_endpoint_types": ["openai", "anthropic", "gemini", "openai-response"]},
        {"id": "text/plain-chat", "supported_endpoint_types": ["openai"], "architecture": {"input_modalities": ["text"]}},
        {
            "id": "vision/chat-vlm",
            "name": "Vision Chat",
            "supported_endpoint_types": ["openai"],
            "architecture": {"input_modalities": ["text", "image"]},
        },
        {"id": "undeclared/chat", "supported_endpoint_types": ["openai"]},
        {"id": "image/img-gen", "supported_endpoint_types": ["image-generation"]},
        {"id": "video/vid-gen", "supported_endpoint_types": ["openai-video"]},
        {"id": "jina/jina-rerank-v3", "supported_endpoint_types": ["jina-rerank"]},
        {"id": "embed/text-embed", "supported_endpoint_types": ["embeddings"]},
        {"id": "orca/orcacyber-ultra-1.0", "supported_endpoint_types": ["openai", "anthropic"]},
        "not-a-record",
    ]
}


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    for name in ("ORCAROUTER_API_KEY", "ORCA_BASE_URL", "ORCA_AUTH_BASE_URL", "ORCA_API_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv(orca.CONFIG_DIR_ENV, str(tmp_path / "config"))
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "cache"))
    OrcaRouterModelCatalog.reset()
    yield
    OrcaRouterModelCatalog.reset()


CACHE_DIR_ENV = "RISE_CACHE_DIR"


def _write_store(**overrides):
    data = {
        "method": "pkce",
        "key": FAKE_KEY,
        "user_id": "12345",
        "scope": "api",
        "api_base": "https://api.orcarouter.ai/v1",
        "generation": 1,
        "needs_reauth": False,
    }
    data.update(overrides)
    orca._secure_write(orca._credentials_path(), data)
    return data


def _store_text():
    return orca._credentials_path().read_text()


# ---------------------------------------------------------------- catalog ---


def test_catalog_parsing_and_text_filter_excludes_non_chat_models():
    catalog = ModelCatalog(models=tuple(parse_catalog_payload(CATALOG_BODY)), source="live")
    text_ids = [m.id for m in catalog.for_capability(CAPABILITY_TEXT)]
    assert "orcarouter/auto" in text_ids
    assert "text/plain-chat" in text_ids
    assert "undeclared/chat" in text_ids  # chat endpoint type alone is enough for text
    assert "orca/orcacyber-ultra-1.0" in text_ids
    for excluded in ("image/img-gen", "video/vid-gen", "jina/jina-rerank-v3", "embed/text-embed"):
        assert excluded not in text_ids


def test_vlm_filter_fails_closed_for_undeclared_modalities():
    catalog = ModelCatalog(models=tuple(parse_catalog_payload(CATALOG_BODY)), source="live")
    vlm_ids = [m.id for m in catalog.for_capability(CAPABILITY_VLM)]
    assert vlm_ids == ["vision/chat-vlm"]


def test_live_discovery_is_authoritative_and_seed_not_mixed():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
        return httpx.Response(200, json=CATALOG_BODY)

    catalog = OrcaRouterModelCatalog(
        "https://api.orcarouter.ai/v1", api_key=FAKE_KEY, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ).fetch()
    assert catalog.source == "live" and not catalog.degraded
    ids = {m.id for m in catalog.models}
    assert "openai/gpt-5.5" not in ids  # the seed must not be merged in
    assert len(calls) == 1


def test_discovery_failure_falls_back_to_verified_seed_with_metadata():
    def handler(request):
        raise httpx.ConnectTimeout("boom")

    catalog = OrcaRouterModelCatalog(
        "https://api.orcarouter.ai/v1", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ).fetch()
    assert catalog.source == "seed" and catalog.degraded
    seed_text = {m.id for m in catalog.for_capability(CAPABILITY_TEXT)}
    assert {"orcarouter/auto", "orcarouter/fusion", "openai/gpt-5.5", "deepseek/deepseek-v4-pro"} <= seed_text
    gpt = next(m for m in catalog.models if m.id == "openai/gpt-5.5")
    assert "image" in gpt.input_modalities
    assert "openai/gpt-5.5" in {m.id for m in catalog.for_capability(CAPABILITY_VLM)}


def test_discovery_auth_error_uses_last_known_good_cache_then_seed():
    first = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=CATALOG_BODY)))
    catalog = OrcaRouterModelCatalog("https://api.orcarouter.ai/v1", http_client=first).fetch()
    assert catalog.source == "live"

    def unauthorized(request):
        return httpx.Response(401, json={"error": "bad key"})

    second = httpx.Client(transport=httpx.MockTransport(unauthorized))
    OrcaRouterModelCatalog.reset()
    recovered = OrcaRouterModelCatalog("https://api.orcarouter.ai/v1", http_client=second).fetch()
    assert recovered.source == "cache"
    assert {m.id for m in recovered.models} == {m.id for m in catalog.models}


def test_cache_from_other_origin_is_not_restored():
    live = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=CATALOG_BODY)))
    OrcaRouterModelCatalog("https://api.orcarouter.ai/v1", http_client=live).fetch()

    def boom(request):
        raise httpx.ConnectError("down")

    OrcaRouterModelCatalog.reset()
    other = OrcaRouterModelCatalog(
        "https://relay.selfhosted.example/v1", http_client=httpx.Client(transport=httpx.MockTransport(boom))
    ).fetch()
    assert other.source == "seed"


def test_catalog_bounds_item_count():
    big = {"data": [{"id": f"m/{i}", "supported_endpoint_types": ["openai"]} for i in range(5000)]}
    from image_red_teaming.clients.orcarouter_catalog import MAX_CATALOG_ITEMS

    assert len(parse_catalog_payload(big)) == MAX_CATALOG_ITEMS
    with pytest.raises(ValueError):
        parse_catalog_payload({"nope": 1})


def test_stale_model_warning():
    catalog = ModelCatalog(models=tuple(parse_catalog_payload(CATALOG_BODY)), source="live")
    assert stale_model_warning(catalog, CAPABILITY_VLM, "text/plain-chat") is not None
    assert stale_model_warning(catalog, CAPABILITY_TEXT, "text/plain-chat") is None


# ---------------------------------------------------------------- origins ---


def test_origins_are_separate_by_default_and_overrides_precede(monkeypatch):
    assert orca.resolve_auth_base() == "https://www.orcarouter.ai"
    assert orca.resolve_api_base() == "https://api.orcarouter.ai/v1"
    monkeypatch.setenv("ORCA_BASE_URL", "https://orca.internal.example")
    assert orca.resolve_auth_base() == "https://orca.internal.example"
    assert orca.resolve_api_base() == "https://orca.internal.example/v1"
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "https://auth.example.test")
    monkeypatch.setenv("ORCA_API_BASE_URL", "http://127.0.0.1:8080/v1")
    assert orca.resolve_auth_base() == "https://auth.example.test"
    assert orca.resolve_api_base() == "http://127.0.0.1:8080/v1"
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "http://evil.example.test")
    with pytest.raises(orca.OrcaCredentialError):
        orca.resolve_auth_base()


def test_authorize_url_uses_auth_path_and_s256():
    url = orca.build_authorize_url(
        "https://www.orcarouter.ai", "CHALLENGE", "STATE", "http://127.0.0.1:5555/cb", "RISE"
    )
    assert url.startswith("https://www.orcarouter.ai/auth?")
    assert "code_challenge_method=S256" in url
    assert "callback_url=http%3A%2F%2F127.0.0.1%3A5555%2Fcb" in url
    assert "/api/v1/auth" not in url and "/v1/auth" not in url


def test_pkce_verifier_challenge_and_state_are_fresh():
    v1, v2 = orca.new_verifier(), orca.new_verifier()
    assert v1 != v2
    challenge = orca.code_challenge_for(v1)
    assert challenge == orca.code_challenge_for(v1)
    assert challenge == orca.base64.urlsafe_b64encode(hashlib.sha256(v1.encode()).digest()).decode().rstrip("=")
    assert "=" not in challenge
    assert orca.new_state() != orca.new_state()
    assert orca.states_match("abc", "abc")
    assert not orca.states_match("abc", "abd")
    assert not orca.states_match("abc", None)


# -------------------------------------------------------------- api key ---


def test_api_key_adapter_reads_env_and_prefers_it_over_store(monkeypatch):
    _write_store()
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-env-key")
    cred = orca.EnvApiKeySource().acquire()
    assert cred.key == "sk-orca-env-key" and cred.source == "api_key"


def test_api_key_adapter_missing_key_is_actionable_and_secret_free(monkeypatch):
    with pytest.raises(orca.OrcaCredentialError) as info:
        orca.EnvApiKeySource().acquire()
    assert "ORCAROUTER_API_KEY" in str(info.value)
    assert "orcarouter-login" in str(info.value)


def test_api_key_adapter_uses_stored_pkce_login_when_env_absent():
    _write_store()
    cred = orca.EnvApiKeySource().acquire()
    assert cred.key == FAKE_KEY and cred.source == "pkce"


def test_api_key_adapter_ignores_store_marked_needs_reauth(monkeypatch):
    _write_store(needs_reauth=True)
    with pytest.raises(orca.OrcaCredentialError):
        orca.EnvApiKeySource().acquire()


def test_credential_repr_masks_key():
    cred = orca.OrcaCredential(key=FAKE_KEY, source="api_key")
    assert FAKE_KEY not in repr(cred) and FAKE_KEY not in str(cred)


def test_redact_removes_key_material():
    assert FAKE_KEY not in orca.redact(f"request failed for {FAKE_KEY}", FAKE_KEY)
    assert "sk-orca-abc123" not in orca.redact("token sk-orca-abc123 leaked")


# ------------------------------------------------------------------ pkce ---


class FakeAuthServer:
    """Local stand-in for www.orcarouter.ai: consent redirect + exchange."""

    def __init__(self, code="code-1", scope="api", status=200):
        self.code = code
        self.scope = scope
        self.status = status
        self.authorize_requests = []
        self.exchange_requests = []
        self.base = ""

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/auth":
                self.authorize_requests.append(request.url)
                redirect = request.url.params["callback_url"]
                browser = httpx.Client(transport=httpx.MockTransport(self._callback))
                browser.get(
                    redirect,
                    params={"code": self.code, "state": request.url.params["state"]},
                )
                return httpx.Response(302, headers={"Location": redirect})
            if request.url.path == "/api/v1/auth/keys":
                body = json.loads(request.content)
                self.exchange_requests.append(body)
                if self.status != 200:
                    return httpx.Response(self.status, json={"error": "x"})
                return httpx.Response(200, json={"key": FAKE_KEY, "user_id": "12345", "scope": self.scope})
            return httpx.Response(404)

        return httpx.Client(transport=httpx.MockTransport(handler))

    def _callback(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")


def _connect(source: orca.PkceKeySource, fake: FakeAuthServer) -> orca.OrcaCredential:
    def deliver(listener, authorize_url):
        fake.authorize_requests.append(httpx.URL(authorize_url))
        params = {"code": fake.code, "state": httpx.URL(authorize_url).params["state"]}
        urllib.request.urlopen(
            listener.callback_url + "?" + urllib.parse.urlencode(params),
            timeout=5,
        ).read()
        return listener.wait_for_code(httpx.URL(authorize_url).params["state"], timeout=5)

    source._callback_delivery = deliver
    source._http = fake.client()
    return source.connect()


def test_pkce_flow_completes_and_persists_with_0600(monkeypatch):
    fake = FakeAuthServer()
    captured = {"logging": []}
    handler = logging.StreamHandler(_Capture(captured["logging"]))
    logging.getLogger("image_red_teaming.clients.orcarouter").addHandler(handler)
    source = orca.PkceKeySource(auth_base="https://www.orcarouter.ai", api_base="https://api.orcarouter.ai/v1")
    cred = _connect(source, fake)
    logging.getLogger("image_red_teaming.clients.orcarouter").removeHandler(handler)

    assert cred.key == FAKE_KEY and cred.source == "pkce" and cred.scope == "api"
    url = fake.authorize_requests[0]
    assert url.host == "www.orcarouter.ai" and url.path == "/auth"
    body = fake.exchange_requests[0]
    assert set(body) == {"code", "code_verifier", "code_challenge_method"}
    assert body["code_challenge_method"] == "S256"
    assert body["code"] == "code-1"
    assert orca.code_challenge_for(body["code_verifier"]) == url.params["code_challenge"]
    assert body["code_verifier"] not in str(url)
    assert FAKE_KEY not in " ".join(captured["logging"])
    assert body["code_verifier"] not in " ".join(captured["logging"])
    stored = json.loads(_store_text())
    assert stored["key"] == FAKE_KEY and stored["generation"] == 1 and stored["method"] == "pkce"
    assert stored["account"] == "user:12345"
    assert oct(os.stat(orca._credentials_path()).st_mode & 0o777) == "0o600"


def test_pkce_reuses_stored_key_instead_of_minting_a_second_one():
    source = orca.PkceKeySource(auth_base="https://www.orcarouter.ai")
    _write_store(generation=3)
    cred = source.acquire()
    assert cred.generation == 3 and cred.key == FAKE_KEY


def test_pkce_state_mismatch_is_rejected():
    fake = FakeAuthServer()
    source = orca.PkceKeySource(auth_base="https://www.orcarouter.ai")

    def deliver(listener, url):
        params = {"code": "attacker-code", "state": "attacker"}
        urllib.request.urlopen(
            listener.callback_url + "?" + urllib.parse.urlencode(params),
            timeout=5,
        ).read()
        return listener.wait_for_code(httpx.URL(url).params["state"], timeout=5)

    source._callback_delivery = deliver
    source._http = fake.client()
    with pytest.raises(orca.OrcaCredentialError, match="state mismatch"):
        source.connect()
    assert fake.exchange_requests == []
    assert not orca._credentials_path().exists()


def test_pkce_denial_is_a_clean_error():
    source = orca.PkceKeySource(auth_base="https://www.orcarouter.ai")
    fake = FakeAuthServer()

    def deliver(listener, url):
        params = {"error": "access_denied", "state": httpx.URL(url).params["state"]}
        urllib.request.urlopen(
            listener.callback_url + "?" + urllib.parse.urlencode(params),
            timeout=5,
        ).read()
        return listener.wait_for_code(httpx.URL(url).params["state"], timeout=5)

    source._callback_delivery = deliver
    source._http = fake.client()
    with pytest.raises(orca.OrcaCredentialError, match="access_denied"):
        source.connect()


def test_pkce_timeout_releases_and_errors():
    source = orca.PkceKeySource(auth_base="https://www.orcarouter.ai", timeout_seconds=0.2)
    with pytest.raises(orca.OrcaCredentialError, match="timed out"):
        source.connect()


@pytest.mark.parametrize(
    "status,message",
    [(403, "rejected the code"), (429, "rate-limited"), (400, "challenge method"), (500, "HTTP 500")],
)
def test_exchange_failures_are_terminal_and_secret_free(status, message):
    attempt = orca.begin_pkce_attempt("https://www.orcarouter.ai", "RISE")
    fake = FakeAuthServer(status=status)
    with pytest.raises(orca.OrcaCredentialError) as info:
        orca.exchange_code("https://www.orcarouter.ai", "code-x", attempt, http_client=fake.client())
    assert message in str(info.value)
    assert attempt.verifier not in str(info.value) and "code-x" not in str(info.value)
    assert not orca._credentials_path().exists()


def test_exchange_network_error_is_terminal():
    attempt = orca.begin_pkce_attempt("https://www.orcarouter.ai", "RISE")

    def boom(request):
        raise httpx.ConnectError("down")

    with pytest.raises(orca.OrcaCredentialError, match="ConnectError"):
        orca.exchange_code(
            "https://www.orcarouter.ai", "code-x", attempt, http_client=httpx.Client(transport=httpx.MockTransport(boom))
        )


def test_scope_downgrade_is_reported_not_ignored():
    attempt = orca.begin_pkce_attempt("https://www.orcarouter.ai", "RISE")
    cred = orca.exchange_code(
        "https://www.orcarouter.ai", "code-x", attempt, http_client=FakeAuthServer(scope="connector").client()
    )
    assert cred.scope == "connector"  # granted scope read back verbatim


def test_exchange_uses_auth_origin_without_api_override(monkeypatch):
    monkeypatch.setenv("ORCA_API_BASE_URL", "https://api.selfhosted.example/v1")
    seen = []
    attempt = orca.begin_pkce_attempt("https://auth.selfhosted.example", "RISE")
    ok = httpx.Response(200, json={"key": FAKE_KEY, "scope": "api"})

    def handler(request):
        seen.append(str(request.url))
        return ok

    orca.exchange_code(
        "https://auth.selfhosted.example", "c", attempt, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert seen == ["https://auth.selfhosted.example/api/v1/auth/keys"]


# -------------------------------------------------------------- 401 / reauth ---


def _client_with(handler, monkeypatch, model="text/plain-chat"):
    catalog_ok = httpx.Response(200, json=CATALOG_BODY)

    def transport(request):
        if request.url.path.endswith("/models"):
            return catalog_ok
        return handler(request)

    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    return orca.OrcaRouterCyberClient(
        model=model,
        base_url="https://api.orcarouter.ai/v1",
        capability=CAPABILITY_TEXT,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: catalog_ok)),
    ), transport


def test_revoked_stored_key_marks_exact_generation_without_refresh(monkeypatch):
    _write_store(generation=2)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    def transport(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=CATALOG_BODY)
        return handler(request)

    # Build the client with a stored PKCE credential, then point the SDK at a 401 server.
    source = orca.PkceKeySource(auth_base="https://www.orcarouter.ai", api_base="https://api.orcarouter.ai/v1")
    from openai import OpenAI

    client = orca.OrcaRouterCyberClient(
        model="text/plain-chat",
        credential=source.load_stored(),
        base_url="https://api.orcarouter.ai/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=CATALOG_BODY))),
    )
    client.client = OpenAI(
        api_key=FAKE_KEY, base_url="https://api.orcarouter.ai/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    with pytest.raises(Exception):
        client.complete([{"role": "user", "content": "hi", "name": None}])
    assert len(calls) == 1  # no retry storm on 401
    stored = json.loads(_store_text())
    assert stored["needs_reauth"] is True
    assert stored["key"] == FAKE_KEY  # never deleted before a successful relogin


def test_stale_401_does_not_pollute_newer_generation():
    _write_store(generation=5, account="user:12345", needs_reauth=False)
    assert orca.mark_needs_reauth(4, "user:12345") is False
    assert orca.mark_needs_reauth(5, "user:99999") is False  # other account
    assert json.loads(_store_text())["needs_reauth"] is False
    assert orca.mark_needs_reauth(5, "user:12345") is True
    assert json.loads(_store_text())["needs_reauth"] is True


def test_client_validates_model_against_capability(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    orca.OrcaRouterCyberClient(
        model="text/plain-chat",
        base_url="https://api.orcarouter.ai/v1",
        capability=CAPABILITY_TEXT,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=CATALOG_BODY))),
    )
    with pytest.raises(RuntimeError, match="not in the OrcaRouter vlm list"):
        orca.OrcaRouterCyberClient(
            model="text/plain-chat",
            base_url="https://api.orcarouter.ai/v1",
            capability=CAPABILITY_VLM,
            http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=CATALOG_BODY))),
        )


def test_model_override_applies_when_configured_model_is_stale(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    client = orca.OrcaRouterCyberClient(
        model="gone/model",
        model_override="vision/chat-vlm",
        base_url="https://api.orcarouter.ai/v1",
        capability=CAPABILITY_VLM,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=CATALOG_BODY))),
    )
    assert client.model == "vision/chat-vlm"


def test_client_complete_returns_openai_shape(monkeypatch):
    seen = {}

    def transport(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=CATALOG_BODY)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "model": "text/plain-chat",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    from openai import OpenAI

    client = orca.OrcaRouterCyberClient(
        model="text/plain-chat",
        base_url="https://api.orcarouter.ai/v1",
        capability=CAPABILITY_TEXT,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=CATALOG_BODY))),
    )
    client.client = OpenAI(
        api_key=FAKE_KEY, base_url="https://api.orcarouter.ai/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    result = client.complete([{"role": "user", "content": "hi", "name": None}])
    assert result["content"] == "hello"
    assert seen["auth"] == f"Bearer {FAKE_KEY}"
    assert seen["body"]["model"] == "text/plain-chat"
    assert result["usage"]["total_tokens"] == 2


def test_build_llm_routes_orcarouter_client(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    calls = {}

    def fake_client(model, config=None, credential_source=None, base_url=None, capability=None, model_override=None, default_headers=None, http_client=None):
        calls.update(model=model, base_url=base_url, capability=capability, source=type(credential_source).__name__)
        return "client"

    monkeypatch.setattr(orca, "OrcaRouterCyberClient", fake_client)
    cfg = ModelConfig(model="orca/orcacyber-ultra-1.0", client="orcarouter", capability="vlm")
    client = orca.build_orcarouter_client(cfg)
    assert client == "client"
    assert calls == {
        "model": "orca/orcacyber-ultra-1.0",
        "base_url": "https://api.orcarouter.ai/v1",
        "capability": "vlm",
        "source": "EnvApiKeySource",
    }
    env_cfg = ModelConfig(model="orca/orcacyber-ultra-1.0", client="orcarouter", api_key_env="MY_ORCA_KEY")
    orca.build_orcarouter_client(env_cfg)
    assert calls["source"] == "EnvApiKeySource"


def test_default_model_config_env_is_not_openrouter(monkeypatch):
    cfg = ModelConfig(model="orcarouter/auto", client="orcarouter")
    assert cfg.api_key_env == "OPENROUTER_API_KEY"  # shared model default unchanged
    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    # With OPENROUTER_API_KEY unset, the OrcaRouter provider must still resolve its key.
    assert not os.getenv("OPENROUTER_API_KEY")
    assert orca.EnvApiKeySource(ORCAROUTER_API_KEY_ENV).acquire().key == FAKE_KEY
    resolved = {}
    real = orca.EnvApiKeySource

    class Recorder(real):
        def __init__(self, env_name=ORCAROUTER_API_KEY_ENV):
            resolved["env_name"] = env_name
            super().__init__(env_name)

    monkeypatch.setattr(orca, "EnvApiKeySource", Recorder)
    client = orca.build_orcarouter_client(cfg)  # catalog falls back to the seed offline
    assert resolved["env_name"] == ORCAROUTER_API_KEY_ENV
    assert client.credential.key == FAKE_KEY
    assert client.model == "orcarouter/auto"


def test_login_lock_released_on_cancel_and_new_attempt_starts():
    first = orca.begin_pkce_attempt("https://www.orcarouter.ai", "RISE")
    assert orca.active_pkce_attempt_id() == first.attempt_id
    assert orca.cancel_pkce_attempt() is True
    assert first.cancelled
    second = orca.begin_pkce_attempt("https://www.orcarouter.ai", "RISE")
    assert second.attempt_id > first.attempt_id
    assert second.verifier != first.verifier and second.state != first.state
    orca.cancel_pkce_attempt()
    assert orca.cancel_pkce_attempt() is False


class _Capture:
    def __init__(self, sink):
        self.sink = sink

    def write(self, text):
        self.sink.append(text)

    def flush(self):
        pass
