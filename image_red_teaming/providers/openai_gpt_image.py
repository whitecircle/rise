from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlopen
from uuid import uuid4

from openai import OpenAI

from image_red_teaming.providers.base import GeneratedImage, ImageGenProvider, GuardrailError

logger = logging.getLogger(__name__)


class OpenAIGPTImageProvider(ImageGenProvider):
    """
    Minimal connector for OpenAI image generations (e.g., gpt-image-1.5).

    Authentication:
      - Uses the OPENAI_API_KEY environment variable by default.
      - Optionally reads OPENAI_BASE_URL for a custom endpoint.
    """

    name = "openai/gpt-image-1.5"

    def __init__(
        self,
        model_id: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        output_dir: str | Path | None = None,
        default_params: Dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id or "gpt-image-1.5"
        self.name = f"openai/{self.model_id}"
        self.output_dir = Path(output_dir) if output_dir else Path("logs/images")
        self.default_params: Dict[str, Any] = dict(default_params or {})
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} environment variable is not set. "
                "Set it before using the OpenAI image provider."
            )
        base_url_env = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or (base_url_env if base_url_env else None),
            max_retries=8,
        )

    def _is_guardrail_error(self, message: str) -> bool:
        lowered = message.lower()
        return any(
            token in lowered
            for token in ("content_policy", "safety", "moderation", "policy violation")
        )

    def _should_download_url(self) -> bool:
        return "dall-e" in self.model_id.lower()

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

    def generate(self, prompt: str, **kwargs: Any) -> List[GeneratedImage]:
        """
        Submit a request to OpenAI image generations and return a normalized list.

        Extra keyword arguments are passed directly into the image generation call, e.g.:
          - size="1024x1024"
          - quality="high"
          - n=2
        """
        params: Dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt,
        }
        params.update(self.default_params)
        if "response_format" in kwargs:
            params["response_format"] = kwargs.pop("response_format")
        if "image_size" in kwargs and "size" not in kwargs:
            params["size"] = kwargs.pop("image_size")
        params.update(kwargs)

        try:
            result = self.client.images.generate(**params)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if self._is_guardrail_error(message):
                logger.warning(
                    "OpenAI guardrail blocked image generation for %s: %s",
                    self.model_id,
                    message,
                )
                raise GuardrailError(self.name, message) from exc
            logger.exception("OpenAI image generation failed")
            raise

        normalized: List[GeneratedImage] = []
        for item in (result.data or []):
            if isinstance(item, dict):
                b64 = item.get("b64_json")
                url = item.get("url")
                raw_item: Any = dict(item)
            else:
                b64 = getattr(item, "b64_json", None)
                url = getattr(item, "url", None)
                if hasattr(item, "model_dump"):
                    raw_item = item.model_dump()
                elif hasattr(item, "__dict__"):
                    raw_item = dict(item.__dict__)
                else:
                    raw_item = {"repr": repr(item)}

            if isinstance(raw_item, dict) and "b64_json" in raw_item:
                raw_item.pop("b64_json", None)

            file_url = None
            vscode_url = None
            local_path = None
            if b64:
                try:
                    image_bytes = base64.b64decode(b64)
                    local_path, file_url, vscode_url = self._persist_image_bytes(image_bytes)
                except Exception:
                    logger.exception("Failed to persist OpenAI image output")
            elif url and self._should_download_url():
                try:
                    with urlopen(url) as resp:  # nosec: OpenAI URLs are trusted outputs
                        image_bytes = resp.read()
                    b64 = base64.b64encode(image_bytes).decode("ascii")
                    local_path, file_url, vscode_url = self._persist_image_bytes(image_bytes)
                except Exception:
                    logger.exception("Failed to download OpenAI image output")

            record: GeneratedImage = GeneratedImage(
                url=local_path or url or "",
                data_base64=b64,
                content_type="image/png",
                provider=self.name,
                extra={
                    "raw": raw_item,
                    "local_path": local_path,
                    "file_url": file_url,
                    "vscode_url": vscode_url,
                },
            )
            normalized.append(record)
        return normalized
