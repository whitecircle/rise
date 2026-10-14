from __future__ import annotations

import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import httpx

from image_red_teaming.providers.base import GeneratedImage, ImageGenProvider, GuardrailError

logger = logging.getLogger(__name__)


class LeonardoProvider(ImageGenProvider):
    """
    Minimal connector for Leonardo.ai image generations.

    Authentication:
      - Uses the LEONARDO_API_KEY environment variable by default.
      - Optionally reads LEONARDO_BASE_URL for a custom endpoint.
    """

    name = "leonardo"

    # Process-wide rate gate: at most one request every `_min_interval_s` seconds
    # (across all threads/instances). Tune via LEONARDO_MIN_INTERVAL_S env var.
    _rate_lock = threading.Lock()
    _next_allowed_ts: float = 0.0

    def _rate_limit(self) -> None:
        min_interval = float(os.getenv("LEONARDO_MIN_INTERVAL_S", "1.0"))
        with LeonardoProvider._rate_lock:
            now = time.monotonic()
            wait = LeonardoProvider._next_allowed_ts - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            LeonardoProvider._next_allowed_ts = now + min_interval

    def __init__(
        self,
        model_id: str | None = None,
        api_key_env: str = "LEONARDO_API_KEY",
        base_url: str | None = None,
        timeout: float = 60.0,
        poll_interval: float = 2.0,
        max_wait_seconds: float = 90.0,
        initial_delay: float = 2.0,
        include_base64: bool = True,
        output_dir: str | Path | None = None,
        save_images: bool = True,
    ) -> None:
        self.model_id = model_id or "7b592283-e8a7-4c5a-9ba6-d18c31f258b9"
        self.name = f"leonardo/{self.model_id}"
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds
        self.initial_delay = initial_delay
        self.include_base64 = include_base64
        self.save_images = save_images
        self.output_dir = Path(output_dir) if output_dir else Path("logs/images")

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} environment variable is not set. "
                "Set it before using the Leonardo provider."
            )
        base_url = base_url or os.getenv("LEONARDO_BASE_URL", "https://cloud.leonardo.ai/api/rest/v1")
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    def _is_guardrail_error(self, message: str) -> bool:
        lowered = message.lower()
        return any(token in lowered for token in ("policy", "safety", "moderation", "nsfw", "blocked"))

    def _extract_error_message(self, payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("error", "message", "detail", "reason"):
                value = payload.get(key)
                if value:
                    return str(value)
            if payload.get("errors"):
                return str(payload["errors"])
        return str(payload)

    def _request_with_retry(self, method: str, url: str, *, max_attempts: int = 6, base_delay: float = 2.0, **kwargs: Any) -> httpx.Response:
        attempt = 0
        while True:
            self._rate_limit()
            response = self._client.request(method, url, **kwargs)
            if response.status_code != 429 or attempt >= max_attempts - 1:
                return response
            retry_after = response.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else base_delay * (2 ** attempt)
            except (TypeError, ValueError):
                delay = base_delay * (2 ** attempt)
            logger.warning("Leonardo 429 on %s %s; sleeping %.1fs (attempt %d/%d)", method, url, delay, attempt + 1, max_attempts)
            time.sleep(delay)
            attempt += 1

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            payload = None
            try:
                payload = response.json()
            except Exception:
                payload = response.text
            message = self._extract_error_message(payload)
            if response.status_code == 403:
                raise GuardrailError(self.name, message or "Leonardo request forbidden") from exc
            if self._is_guardrail_error(message):
                raise GuardrailError(self.name, message) from exc
            raise

    def _extract_generation_id(self, payload: Dict[str, Any]) -> str | None:
        job = payload.get("sdGenerationJob") or payload.get("generationJob") or {}
        for key in ("generationId", "generationID", "id"):
            value = job.get(key)
            if value:
                return str(value)
        for key in ("generationId", "generationID", "id"):
            value = payload.get(key)
            if value:
                return str(value)
        return None

    def _extract_images_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "generations_by_pk" in payload and isinstance(payload["generations_by_pk"], dict):
            images = payload["generations_by_pk"].get("generated_images") or payload["generations_by_pk"].get("images")
        else:
            images = payload.get("generated_images") or payload.get("images")
        if not images:
            return []
        if isinstance(images, dict):
            return [images]
        if isinstance(images, list):
            return [img for img in images if isinstance(img, dict)]
        return []

    def _extract_status(self, payload: Dict[str, Any]) -> str:
        if "generations_by_pk" in payload and isinstance(payload["generations_by_pk"], dict):
            status = payload["generations_by_pk"].get("status")
            if status:
                return str(status)
        status = payload.get("status") or payload.get("generationStatus")
        return str(status or "")

    def _persist_image_bytes(
        self, image_bytes: bytes
    ) -> tuple[str | None, str | None, str | None]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            filename = f"{self.model_id.replace('/', '_')}_{uuid4().hex}.png"
            path = self.output_dir / filename
            try:
                with path.open("xb") as f:
                    f.write(image_bytes)
                resolved = path.resolve()
                file_url = resolved.as_uri()
                vscode_url = f"vscode://file{resolved.as_posix()}"
                return str(resolved), file_url, vscode_url
            except FileExistsError as exc:
                logger.warning("Image filename collision at %s: %s", path, exc)
                continue
        logger.warning("Failed to allocate unique image filename after retries")
        return None, None, None

    def _download_image_bytes(self, url: str) -> tuple[bytes | None, str | None]:
        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "image/png")
            return resp.content, content_type
        except Exception:
            logger.exception("Failed to download Leonardo image", extra={"url": url})
            return None, None

    def _normalize_images(self, images: List[Dict[str, Any]]) -> List[GeneratedImage]:
        normalized: List[GeneratedImage] = []
        for img in images:
            url = img.get("url") or img.get("imageUrl") or img.get("uri")
            if not url:
                continue
            data_base64 = None
            content_type = img.get("content_type") or img.get("mimeType") or "image/png"
            local_path = None
            file_url = None
            vscode_url = None
            if self.include_base64 or self.save_images:
                image_bytes, downloaded_type = self._download_image_bytes(url)
                if image_bytes:
                    if self.include_base64:
                        data_base64 = base64.b64encode(image_bytes).decode("ascii")
                    if self.save_images:
                        local_path, file_url, vscode_url = self._persist_image_bytes(image_bytes)
                if downloaded_type:
                    content_type = downloaded_type
            record: GeneratedImage = GeneratedImage(
                url=url,
                data_base64=data_base64,
                width=img.get("width"),  # type: ignore[arg-type]
                height=img.get("height"),  # type: ignore[arg-type]
                content_type=content_type,
                provider=self.name,
                extra={
                    "raw": img,
                    "local_path": local_path,
                    "file_url": file_url,
                    "vscode_url": vscode_url,
                },
            )
            normalized.append(record)
        return normalized

    def _poll_generation(self, generation_id: str) -> List[GeneratedImage]:
        start = time.monotonic()
        while True:
            response = self._request_with_retry("GET", f"generations/{generation_id}")
            self._raise_for_status(response)
            payload = response.json()
            images_payload = self._extract_images_payload(payload)
            if images_payload:
                return self._normalize_images(images_payload)

            status = self._extract_status(payload).lower()
            if any(token in status for token in ("fail", "error", "cancel", "blocked")):
                message = self._extract_error_message(payload)
                if self._is_guardrail_error(message):
                    raise GuardrailError(self.name, message)
                raise RuntimeError(f"Leonardo generation failed: {message}")

            if self.max_wait_seconds and (time.monotonic() - start) > self.max_wait_seconds:
                raise TimeoutError(f"Timed out waiting for Leonardo generation {generation_id}")

            time.sleep(self.poll_interval)

    def generate(self, prompt: str, **kwargs: Any) -> List[GeneratedImage]:
        """
        Submit a request to Leonardo.ai and return a normalized list of images.

        Extra keyword arguments are passed into the generation request, e.g.:
          - height=512
          - width=512
          - num_images=1
        """
        payload: Dict[str, Any] = {"prompt": prompt, "modelId": self.model_id}

        height = kwargs.pop("height", None)
        width = kwargs.pop("width", None)
        image_size = kwargs.pop("image_size", None)
        if image_size and (height is None or width is None) and isinstance(image_size, str):
            if "x" in image_size:
                size_parts = image_size.lower().split("x", 1)
                if len(size_parts) == 2:
                    try:
                        width = width or int(size_parts[0])
                        height = height or int(size_parts[1])
                    except ValueError:
                        logger.warning("Could not parse image_size=%s", image_size)

        num_images = kwargs.pop("num_images", None)
        if num_images is None:
            num_images = kwargs.pop("numImages", None)
        if num_images is None:
            num_images = kwargs.pop("n", None)
        if num_images is None:
            num_images = 1

        if height is None and width is None:
            height = 512
            width = 512
        elif height is None and width is not None:
            height = width
        elif width is None and height is not None:
            width = height

        model_override = kwargs.pop("model_id", None)
        if model_override and "modelId" not in kwargs:
            kwargs["modelId"] = model_override

        if height is not None:
            payload["height"] = height
        if width is not None:
            payload["width"] = width
        if num_images is not None:
            payload["num_images"] = num_images

        payload.update(kwargs)

        response = self._request_with_retry("POST", "generations", json=payload)
        self._raise_for_status(response)
        data = response.json()
        generation_id = self._extract_generation_id(data)
        if not generation_id:
            raise RuntimeError(f"Leonardo response missing generationId: {data}")

        if self.initial_delay:
            time.sleep(self.initial_delay)

        return self._poll_generation(generation_id)
