"""
OrcaRouter Cyber Inference provider.

OrcaRouter is an OpenAI-compatible AI gateway for security teams: red-team
workloads (authorized, scoped testing) and blue-team telemetry can share one
endpoint.  Inference and model discovery use ``https://api.orcarouter.ai/v1``;
authentication (account sign-in) lives on ``https://www.orcarouter.ai``.

Two ways to obtain the credential are offered, and are kept here behind one
small interface so the provider, the catalog and every AI input entry point
share a single credential path:

- :class:`EnvApiKeySource` - paste an ``sk-orca-...`` key into the existing
  ``.env`` / environment (``ORCAROUTER_API_KEY``);
- :class:`PkceKeySource` - "Connect with OrcaRouter" via OAuth 2.0 + PKCE
  (loopback redirect, refresh-unsafe key grants are swapped for a cheap probe).

Both return the same :class:`OrcaCredential`.  The resolved value is a normal
OrcaRouter API key and is reused until OrcaRouter revokes it; there is no
refresh grant, so a ``401`` marks the credential for re-authentication.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from image_red_teaming.clients.base import ChatMessage, CompletionResult, LLMClient
from image_red_teaming.clients.orcarouter_catalog import (
    ORCAROUTER_API_KEY_ENV,
    ORCAROUTER_DEFAULT_API_BASE,
    ORCAROUTER_PROVIDER_NAME,
)

logger = logging.getLogger(__name__)

# Authentication lives on its own public origin (never derived from the API base).
ORCAROUTER_AUTH_BASE = "https://www.orcarouter.ai"

# ---------------------------------------------------------------------------
# Credential seam
# ---------------------------------------------------------------------------

KEY_PREFIX = "sk-orca-"
AUTHORIZE_PATH = "/auth"
EXCHANGE_PATH = "/api/v1/auth/keys"
CONFIG_DIR_ENV = "RISE_CONFIG_DIR"
CREDENTIALS_FILENAME = "orcarouter.json"

MASK = "sk-orca-****"


class OrcaCredentialError(RuntimeError):
    """Raised when a credential cannot be obtained.  Never contains a secret."""


class OrcaReauthRequired(OrcaCredentialError):
    """Terminal 401 from the relay: the current credential generation is dead."""


class OrcaCredential:
    """A usable OrcaRouter API key and the account generation it belongs to."""

    def __init__(
        self,
        key: str,
        source: str,
        generation: int = 1,
        account: Optional[str] = None,
        scope: Optional[str] = None,
        needs_reauth: bool = False,
    ) -> None:
        if not key:
            raise OrcaCredentialError("OrcaRouter credential is empty")
        self._key = key
        self.source = source  # "api_key" | "pkce"
        # (account, generation) identifies the exact credential a request used,
        # so a late 401 from an old request cannot mark a newer login as broken.
        self.account = account or f"{source}:default"
        self.generation = generation
        self.scope = scope
        self.needs_reauth = needs_reauth

    @property
    def key(self) -> str:
        return self._key

    def __repr__(self) -> str:  # never leak the key
        return (
            f"OrcaCredential(source={self.source!r}, account={self.account!r}, "
            f"generation={self.generation}, needs_reauth={self.needs_reauth})"
        )

    __str__ = __repr__

    def masked(self) -> str:
        return MASK


class CredentialSource(ABC):
    """How a credential is obtained.  Two adapters: API key and PKCE."""

    method: str

    @abstractmethod
    def acquire(self) -> OrcaCredential:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


def _config_dir() -> Path:
    root = os.getenv(CONFIG_DIR_ENV)
    if root:
        return Path(root)
    xdg = os.getenv("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "rise"


def _credentials_path() -> Path:
    return _config_dir() / CREDENTIALS_FILENAME


def _secure_write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    tmp.replace(path)


def _read_credentials() -> Optional[Dict[str, Any]]:
    path = _credentials_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        logger.debug("Ignoring unreadable OrcaRouter credentials file", exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _looks_like_key(value: str) -> bool:
    return bool(value) and value.startswith(KEY_PREFIX)


def redact(text: str, secret: Optional[str] = None) -> str:
    """Remove key material from anything that may be logged."""
    if secret:
        text = text.replace(secret, MASK)
    return re.sub(re.escape(KEY_PREFIX) + r"[A-Za-z0-9_\-]+", MASK, text)


class EnvApiKeySource(CredentialSource):
    """API-key adapter: read ``ORCAROUTER_API_KEY`` from env / .env."""

    method = "api_key"

    def __init__(self, env_name: str = ORCAROUTER_API_KEY_ENV) -> None:
        self.env_name = env_name
        self._credential: Optional[OrcaCredential] = None

    def acquire(self) -> OrcaCredential:
        stored = _read_credentials() or {}
        value = (os.getenv(self.env_name) or "").strip()
        if not value:
            if self._stored_key_usable(stored):
                return OrcaCredential(
                    key=stored["key"],
                    source=PkceKeySource.method,
                    generation=int(stored.get("generation") or 1),
                    account=stored.get("account"),
                    scope=stored.get("scope"),
                )
            raise_orca_missing_key(self.env_name)
        if not _looks_like_key(value):
            logger.warning(
                "%s does not start with '%s'. Treating it as an OrcaRouter key anyway; "
                "an invalid key will surface as HTTP 401 on the first request.",
                self.env_name,
                KEY_PREFIX,
            )
        # Keep .env precedence: the environment always wins over a stored login.
        self._credential = OrcaCredential(
            key=value,
            source=self.method,
            generation=int(stored.get("generation") or 1),
            account=stored.get("account") or f"env:{self.env_name}",
            scope=stored.get("scope") or "api",
        )
        return self._credential

    @staticmethod
    def _stored_key_usable(stored: Dict[str, Any]) -> bool:
        """A connected account's key is a fallback when the env var is absent."""
        return (
            stored.get("method") == PkceKeySource.method
            and isinstance(stored.get("key"), str)
            and bool(stored.get("key"))
            and not stored.get("needs_reauth")
        )

    def clear(self) -> None:
        """Remove a stored PKCE login fallback for this provider (env is user-owned)."""
        self._credential = None


class PkceKeySource(CredentialSource):
    """
    PKCE adapter: "Connect with OrcaRouter".

    Flow A (loopback redirect) is used because RISE runs as a local CLI whose
    user has a browser: the callback is delivered to a one-shot 127.0.0.1
    listener.  The returned key is stored with 0600 permissions and reused on
    every later launch; nothing is refreshed.
    """

    method = "pkce"

    def __init__(
        self,
        auth_base: Optional[str] = None,
        api_base: Optional[str] = None,
        app_name: str = "RISE",
        timeout_seconds: float = 300.0,
        http_client: Optional[httpx.Client] = None,
        on_authorize_url: Optional[Callable[[str], None]] = None,
        open_browser: bool = True,
        callback_delivery: Optional[Callable[["_LoopbackListener", str], str]] = None,
    ) -> None:
        self.auth_base = (auth_base or resolve_auth_base()).rstrip("/")
        self.api_base = (api_base or resolve_api_base()).rstrip("/")
        self.app_name = app_name
        self.timeout_seconds = timeout_seconds
        self._http = http_client
        self._on_authorize_url = on_authorize_url or _default_print_url
        self._open_browser = open_browser
        self._callback_delivery = callback_delivery

    # -- credentials -------------------------------------------------------
    def acquire(self, force_new: bool = False) -> OrcaCredential:
        if not force_new:
            stored = self.load_stored()
            if stored is not None and not stored.needs_reauth:
                return stored
            if stored is not None:
                logger.warning(
                    "OrcaRouter credential needs re-authentication; starting a new login "
                    "(the old key is kept until it succeeds)."
                )
        return self.connect()

    def load_stored(self) -> Optional[OrcaCredential]:
        data = _read_credentials()
        if not data or data.get("method") != self.method or not data.get("api_base"):
            return None
        if data.get("api_base") != self.api_base:
            return None
        key = data.get("key")
        if not isinstance(key, str) or not key:
            return None
        return OrcaCredential(
            key=key,
            source=self.method,
            generation=int(data.get("generation") or 1),
            account=data.get("account"),
            scope=data.get("scope"),
            needs_reauth=bool(data.get("needs_reauth")),
        )

    def connect(self) -> OrcaCredential:
        previous = _read_credentials() or {}
        attempt = begin_pkce_attempt(self.auth_base, self.app_name, self.timeout_seconds)
        try:
            self._on_authorize_url(attempt.authorize_url)
            code = self._wait_for_code(attempt)
            credential = exchange_code(
                self.auth_base, code, attempt, http_client=self._http,
            )
        finally:
            attempt.listener.close()
        generation = int(previous.get("generation") or 0) + 1
        credential.generation = generation
        # Persist only after a successful exchange; the old secret stays until then.
        _secure_write(
            _credentials_path(),
            {
                "method": self.method,
                "key": credential.key,
                "account": credential.account,
                "scope": credential.scope,
                "api_base": self.api_base,
                "auth_base": self.auth_base,
                "generation": generation,
                "needs_reauth": False,
                "created_at": int(time.time()),
            },
        )
        return credential

    def clear(self) -> None:
        path = _credentials_path()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Could not remove OrcaRouter credentials file %s", path)

    # -- flow --------------------------------------------------------------
    def _wait_for_code(self, attempt: "PkceAttempt") -> str:
        if self._callback_delivery is not None:
            return self._callback_delivery(attempt.listener, attempt.authorize_url)
        if self._open_browser:
            _open_in_browser(attempt.authorize_url)
        return attempt.listener.wait_for_code(attempt.state, timeout=attempt.timeout)


def raise_orca_missing_key(env_name: str) -> None:
    raise OrcaCredentialError(
        f"{env_name} is not set for client 'orcarouter'. Set it in .env (create a key at "
        f"https://www.orcarouter.ai/console), or run "
        f"`python main.py --config <cfg> orcarouter-login` to connect with your account."
    )


def _default_print_url(url: str) -> None:
    print(
        "Opening your browser to authorize RISE with OrcaRouter.\n"
        f"If it does not open, visit:\n  {url}",
        flush=True,
    )


def _open_in_browser(url: str) -> None:  # noqa: D401
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception:
        logger.warning("Could not open a browser. Visit the printed URL manually.")


# ---------------------------------------------------------------------------
# Origin policy
# ---------------------------------------------------------------------------

def _is_loopback_host(host: str) -> bool:
    return host in ("localhost", "127.0.0.1", "::1", "[::1]")


def _enforce_origin(url: str, name: str) -> str:
    parsed = httpx.URL(url)
    scheme = parsed.scheme
    if scheme == "https":
        return url.rstrip("/")
    if scheme == "http" and _is_loopback_host(parsed.host):
        return url.rstrip("/")
    raise OrcaCredentialError(f"{name} must be HTTPS (HTTP is allowed only for loopback): {url}")


def resolve_auth_base() -> str:
    """Auth origin: explicit override > shared self-hosted base > public default."""
    explicit = os.getenv("ORCA_AUTH_BASE_URL")
    if explicit:
        return _enforce_origin(explicit, "ORCA_AUTH_BASE_URL")
    shared = os.getenv("ORCA_BASE_URL")
    if shared:
        return _enforce_origin(shared, "ORCA_BASE_URL")
    return ORCAROUTER_AUTH_BASE


def resolve_api_base() -> str:
    """Inference origin: explicit override > shared self-hosted base > public default."""
    explicit = os.getenv("ORCA_API_BASE_URL")
    if explicit:
        return _enforce_origin(explicit, "ORCA_API_BASE_URL")
    shared = os.getenv("ORCA_BASE_URL")
    if shared:
        base = _enforce_origin(shared, "ORCA_BASE_URL")
        return base if base.endswith("/v1") else f"{base}/v1"
    return ORCAROUTER_DEFAULT_API_BASE


def build_authorize_url(
    auth_base: str,
    code_challenge: str,
    state: str,
    callback_url: str,
    app_name: str,
    scope: str = "api",
) -> str:
    from urllib.parse import urlencode

    query = urlencode(
        {
            "callback_url": callback_url,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "app_name": app_name,
            "scope": scope,
        }
    )
    return f"{auth_base.rstrip('/')}{AUTHORIZE_PATH}?{query}"


# ---------------------------------------------------------------------------
# PKCE primitives
# ---------------------------------------------------------------------------

def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def new_verifier() -> str:
    return b64url(os.urandom(32))


def code_challenge_for(verifier: str) -> str:
    return b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def new_state() -> str:
    return b64url(os.urandom(16))


def states_match(expected: str, received: Optional[str]) -> bool:
    import hmac

    return isinstance(received, str) and bool(expected) and hmac.compare_digest(
        expected.encode("utf-8"), received.encode("utf-8")
    )


_PKCE_ATTEMPT_LOCK = threading.Lock()


class _LoopbackListener:
    """One-shot HTTP listener on 127.0.0.1 that captures the redirect."""

    def __init__(self) -> None:
        import http.server
        import socketserver
        import threading as _threading

        self._event = _threading.Event()
        self._query: Optional[Dict[str, str]] = None
        listener = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                from urllib.parse import parse_qs, urlparse

                parsed = urlparse(self.path)
                if parsed.path != "/cb":
                    self.send_response(404)
                    self.end_headers()
                    return
                listener._query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                body = b"<p>Connected to OrcaRouter. You can close this tab.</p>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                listener._event.set()

            def log_message(self, *args: Any) -> None:  # keep the callback quiet
                return

        self._server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = _threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def callback_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/cb"

    def wait_for_code(self, expected_state: str, timeout: float = 300.0) -> str:
        if not self._event.wait(timeout):
            raise OrcaCredentialError("OrcaRouter authorization timed out; run the login again.")
        query = self._query or {}
        if not states_match(expected_state, query.get("state")):
            raise OrcaCredentialError("OrcaRouter callback state mismatch; authorization rejected.")
        if query.get("error"):
            raise OrcaCredentialError(f"OrcaRouter authorization was not granted ({query['error']}).")
        code = query.get("code")
        if not code:
            raise OrcaCredentialError("OrcaRouter callback did not include a code.")
        return code

    def close(self) -> None:
        try:
            self._server.shutdown()
        finally:
            self._server.server_close()


@dataclass
class PkceAttempt:
    verifier: str
    state: str
    challenge: str
    listener: _LoopbackListener
    authorize_url: str
    timeout: float
    attempt_id: int = 0
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True
        self.listener.close()


_ACTIVE_ATTEMPTS: Dict[int, PkceAttempt] = {}
_ATTEMPT_COUNTER = [0]
_CURRENT_ATTEMPT = [0]


def begin_pkce_attempt(auth_base: str, app_name: str, timeout: float = 300.0) -> PkceAttempt:
    """Create a fresh verifier/state and a loopback listener (Flow A)."""
    listener = _LoopbackListener()
    verifier = new_verifier()
    state = new_state()
    challenge = code_challenge_for(verifier)
    url = build_authorize_url(auth_base, challenge, state, listener.callback_url, app_name)
    with _PKCE_ATTEMPT_LOCK:
        _ATTEMPT_COUNTER[0] += 1
        attempt = PkceAttempt(
            verifier=verifier,
            state=state,
            challenge=challenge,
            listener=listener,
            authorize_url=url,
            timeout=timeout,
            attempt_id=_ATTEMPT_COUNTER[0],
        )
        _ACTIVE_ATTEMPTS[attempt.attempt_id] = attempt
        _CURRENT_ATTEMPT[0] = attempt.attempt_id
    return attempt


def cancel_pkce_attempt(attempt_id: Optional[int] = None) -> bool:
    """Release the login lock: cancel the current (or a given) attempt."""
    with _PKCE_ATTEMPT_LOCK:
        target = attempt_id if attempt_id is not None else _CURRENT_ATTEMPT[0]
        attempt = _ACTIVE_ATTEMPTS.pop(target, None)
        if attempt is None:
            return False
    attempt.cancel()
    return True


def active_pkce_attempt_id() -> int:
    return _CURRENT_ATTEMPT[0]


def _finish_attempt(attempt: PkceAttempt) -> None:
    with _PKCE_ATTEMPT_LOCK:
        _ACTIVE_ATTEMPTS.pop(attempt.attempt_id, None)


def exchange_code(
    auth_base: str,
    code: str,
    attempt: PkceAttempt,
    http_client: Optional[httpx.Client] = None,
) -> OrcaCredential:
    """Redeem an auth code.  Auth origin only; never raises with secrets in text."""
    if attempt.cancelled:
        raise OrcaCredentialError("OrcaRouter authorization was cancelled.")
    url = f"{auth_base.rstrip('/')}{EXCHANGE_PATH}"
    client = http_client or httpx.Client(timeout=30.0, trust_env=False)
    try:
        resp = client.post(
            url,
            json={"code": code, "code_verifier": attempt.verifier, "code_challenge_method": "S256"},
        )
    except httpx.HTTPError as exc:
        raise OrcaCredentialError(
            f"OrcaRouter authorization exchange failed ({type(exc).__name__}); try again."
        ) from None
    finally:
        _finish_attempt(attempt)
        if http_client is None:
            client.close()
    if resp.status_code == 429:
        raise OrcaCredentialError(
            "OrcaRouter rate-limited the sign-in (429). You already hold a stored key; "
            "reuse it or wait and try again."
        )
    if resp.status_code == 403:
        raise OrcaCredentialError(
            "OrcaRouter rejected the code (unknown, expired, already used, or verifier mismatch). Run the login again."
        )
    if resp.status_code == 400:
        raise OrcaCredentialError("OrcaRouter rejected the code challenge method (400).")
    if resp.status_code >= 400:
        raise OrcaCredentialError(f"OrcaRouter authorization exchange failed (HTTP {resp.status_code}).")
    try:
        data = resp.json()
    except ValueError:
        raise OrcaCredentialError("OrcaRouter exchange returned a non-JSON body.") from None
    key = data.get("key") if isinstance(data, dict) else None
    if not isinstance(key, str) or not key:
        raise OrcaCredentialError("OrcaRouter exchange response did not include a key.")
    scope = data.get("scope")
    if scope not in (None, "api", "connector"):
        raise OrcaCredentialError(f"OrcaRouter granted an unexpected scope ({redact(str(scope))}).")
    if scope == "connector":
        logger.warning(
            "OrcaRouter granted scope 'connector', not 'api'. Inference may be limited; "
            "sign in again from a workspace role that can grant 'api'."
        )
    account = str(data.get("user_id") or data.get("account") or "") or None
    return OrcaCredential(key=key, source="pkce", account=(f"user:{account}" if account else None), scope=scope)


def mark_needs_reauth(generation: Optional[int], account: Optional[str] = None) -> bool:
    """
    Terminal 401 handling: mark exactly the rejected account + generation.

    Never deletes the stored key and never attempts a refresh.  Returns False if
    a newer login (different generation or account) has already replaced the
    credential, so a stale failure cannot poison a fresh one.
    """
    data = _read_credentials()
    if not data:
        return False
    current = int(data.get("generation") or 1)
    if generation is not None and generation != current:
        return False
    if account is not None and data.get("account") and data.get("account") != account:
        return False
    data["needs_reauth"] = True
    _secure_write(_credentials_path(), data)
    return True


# ---------------------------------------------------------------------------
# Provider client
# ---------------------------------------------------------------------------

def _is_auth_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 401


class OrcaRouterCyberClient(LLMClient):
    """
    OrcaRouter Cyber Inference chat/VLM client (OpenAI-compatible wire format).

    The model id is validated against the live ``/v1/models`` catalog for the
    client's entry-point capability before the first request is sent.
    """

    name = "orcarouter"
    display_name = ORCAROUTER_PROVIDER_NAME

    def __init__(
        self,
        model: str,
        config: Any = None,
        credential: Optional[OrcaCredential] = None,
        credential_source: Optional[CredentialSource] = None,
        base_url: Optional[str] = None,
        capability: Optional[str] = None,
        model_override: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        super().__init__()
        # Imported here so a catalog import cycle never blocks client construction.
        from image_red_teaming.clients import orcarouter_catalog

        OrcaRouterModelCatalog = orcarouter_catalog.OrcaRouterModelCatalog
        resolve_model_selection = orcarouter_catalog.resolve_model_selection
        CAPABILITY_TEXT = orcarouter_catalog.CAPABILITY_TEXT

        self.config = config
        self.capability = capability or CAPABILITY_TEXT
        self.api_base = (base_url or resolve_api_base()).rstrip("/")
        self._credential = credential
        self._credential_source = credential_source
        source = credential_source or EnvApiKeySource()
        self._credential = credential or source.acquire()

        catalog = OrcaRouterModelCatalog(self.api_base, api_key=self._credential.key, http_client=http_client)
        self.catalog = catalog.fetch()
        self.catalog_source = self.catalog.source
        if self.catalog.degraded:
            logger.warning("OrcaRouter model catalog: %s", self.catalog.detail)
        resolved = resolve_model_selection(
            model,
            catalog,
            self.capability,
            default_model=model,
            model_override=model_override,
        )
        for note in resolved.notes:
            logger.warning(note)
        if resolved.warning:
            raise RuntimeError(f"OrcaRouter provider: {resolved.warning}")
        self.model = resolved.model_id
        default_headers = dict(default_headers or {})
        self.client = OpenAI(
            api_key=self._credential.key,
            base_url=self.api_base,
            default_headers=default_headers or None,
        )

    @property
    def credential(self) -> OrcaCredential:
        return self._credential

    def handle_auth_failure(self) -> None:
        """Called on 401: mark the exact generation, do not retry or refresh."""
        if self._credential.generation and self._credential.source == PkceKeySource.method:
            if mark_needs_reauth(self._credential.generation, self._credential.account):
                self._credential.needs_reauth = True
        logger.error(
            "OrcaRouter rejected the %s credential (%s). Re-run "
            "`python main.py --config <cfg> orcarouter-login --force` to re-authorize.",
            self._credential.source,
            OrcaRouterCyberClient.display_name,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_not_exception_type(OrcaReauthRequired),
    )
    def complete(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResult:
        messages = [{"role": x["role"], "content": x["content"]} for x in messages]
        extra_body = getattr(self, "extra_body", None) or {}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body=extra_body if extra_body else None,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised after classification
            if _is_auth_error(exc):
                # Terminal: never re-sample a revoked credential.
                self.handle_auth_failure()
                raise OrcaReauthRequired(
                    "OrcaRouter rejected the credential (HTTP 401); re-authenticate instead of retrying."
                ) from exc
            raise

        raw: Dict[str, Any] = response.model_dump() if hasattr(response, "model_dump") else {"response": str(response)}
        usage_totals = self._track_usage(raw.get("usage") or {})
        first_choice = response.choices[0]
        content = first_choice.message.content
        reasoning = getattr(first_choice.message, "reasoning_content", None) or getattr(
            first_choice.message, "reasoning", None
        )
        if not reasoning:
            raw_msg = raw.get("choices", [{}])[0].get("message", {})
            reasoning = raw_msg.get("reasoning_content") or raw_msg.get("reasoning")
        if reasoning and isinstance(reasoning, str):
            think_block = f" thinking\n{reasoning}\n</think>\n\n"
            content = think_block + content if content else think_block
        return CompletionResult(
            content=content,
            model=raw.get("model") or self.model,
            usage=usage_totals,
            raw=raw,
        )


def build_orcarouter_client(config: Any, headers: Optional[Dict[str, str]] = None) -> OrcaRouterCyberClient:
    """Build the client for a ``client: orcarouter`` ModelConfig."""
    from image_red_teaming.clients.orcarouter_catalog import CAPABILITY_TEXT, CAPABILITY_VLM

    capability = getattr(config, "capability", None) or CAPABILITY_TEXT
    if capability not in (CAPABILITY_TEXT, CAPABILITY_VLM):
        raise ValueError(f"Unsupported OrcaRouter capability: {capability!r}")
    env_name = getattr(config, "api_key_env", None)
    if not env_name or env_name == "OPENROUTER_API_KEY":
        # ModelConfig's default env name belongs to OpenRouter; never read it here.
        env_name = ORCAROUTER_API_KEY_ENV
    return OrcaRouterCyberClient(
        model=config.model,
        config=config,
        credential_source=EnvApiKeySource(env_name),
        base_url=config.base_url or resolve_api_base(),
        capability=capability,
        model_override=getattr(config, "model_override", None),
        default_headers=headers or config.extra_headers,
    )


def stored_credentials_mode() -> Tuple[str, Optional[Path]]:
    return ("env:" + ORCAROUTER_API_KEY_ENV if os.getenv(ORCAROUTER_API_KEY_ENV) else "pkce-store", _credentials_path())
