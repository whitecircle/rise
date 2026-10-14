from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlopen
from uuid import uuid4

import fal_client
import logging

from image_red_teaming.providers.base import GeneratedImage, ImageGenProvider, GuardrailError

logger = logging.getLogger(__name__)


class FalProvider(ImageGenProvider):
    """
    Minimal connector for image models served by fal.ai.

    Authentication:
      - Uses the `FAL_KEY` environment variable (Fal's default).
    """

    name = "fal"

    def __init__(
        self,
        model_id: str | None = None,
        fal_log_level: int = logging.WARNING,
        output_dir: str | Path | None = None,
        save_images: bool = True,
    ) -> None:
        self.model_id = model_id or "fal-ai/nano-banana-2"
        self.output_dir = Path(output_dir) if output_dir else Path("logs/images")
        self.save_images = save_images
        if not os.getenv("FAL_KEY"):
            # Fail fast to avoid confusing auth errors later.
            raise RuntimeError(
                "FAL_KEY environment variable is not set. "
                "Get an API key from Fal.ai and export FAL_KEY before using this provider."
            )

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

    def generate(
        self, prompt: str, **kwargs: Any
    ) -> List[GeneratedImage]:
        """
        Submit a request to Fal and return a normalized list of images.

        Extra keyword arguments are passed directly into `arguments` for the model, e.g.:
          - image_size="square_hd"
          - seed=123
          - num_images=4

        Returns images with:
          - `url`: Fal-hosted image URL
          - `data_base64`: base64-encoded image bytes (if download succeeds)
        """
        arguments: Dict[str, Any] = {"prompt": prompt}
        arguments.update(kwargs)

        try:
            handler = fal_client.submit(self.model_id, arguments=arguments)
            result = handler.get()
        except fal_client.client.FalClientHTTPError as exc:  # type: ignore[attr-defined]
            logger.warning("Fal guardrail triggered for model %s: %s", self.model_id, exc)
            raise GuardrailError(self.name, str(exc)) from exc

        # Fal responses for image models usually contain an `images` list with url/size.
        images = result.get("images") or []

        normalized: List[GeneratedImage] = []
        for img in images:
            url = img.get("url")
            if not url:
                continue

            data_base64: str | None = None
            local_path: str | None = None
            file_url: str | None = None
            vscode_url: str | None = None
            content_type = img.get("content_type") or ""
            try:
                with urlopen(url) as resp:  # nosec: Fal URLs are trusted outputs
                    raw_bytes = resp.read()
                    if not content_type:
                        try:
                            content_type = resp.headers.get_content_type()
                        except Exception:
                            content_type = ""
                data_base64 = base64.b64encode(raw_bytes).decode("ascii")
                if self.save_images:
                    local_path, file_url, vscode_url = self._persist_image_bytes(raw_bytes)
            except Exception:
                logger.exception("Failed to download or encode image from Fal URL", extra={"url": url})
                data_base64 = None

            if not content_type:
                content_type = "image/png"

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

            # logger.info("Fal image generated: %s", record)

            normalized.append(record)

        return normalized
