"""Provider that calls a local Flux2 inference server (scripts/flux2_server.py)."""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List
from uuid import uuid4

from image_red_teaming.providers.base import GeneratedImage, GuardrailError, ImageGenProvider

logger = logging.getLogger(__name__)


class Flux2LocalProvider(ImageGenProvider):
    """HTTP client for the local Flux2 server."""

    name = "flux2-local"

    def __init__(
        self,
        port: int = 5000,
        host: str = "127.0.0.1",
        output_dir: str | Path | None = None,
        save_images: bool = True,
        num_inference_steps: int = 4,
        guidance_scale: float = 4.0,
        timeout: float = 1800.0,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.output_dir = Path(output_dir) if output_dir else Path("logs/images")
        self.save_images = save_images
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.timeout = timeout

    def _persist_image_bytes(
        self, image_bytes: bytes,
    ) -> tuple[str | None, str | None, str | None]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            filename = f"flux2_{uuid4().hex}.png"
            path = self.output_dir / filename
            try:
                with path.open("xb") as f:
                    f.write(image_bytes)
                resolved = path.resolve()
                return str(resolved), resolved.as_uri(), f"vscode://file{resolved.as_posix()}"
            except FileExistsError:
                continue
        logger.warning("Failed to allocate unique image filename after retries")
        return None, None, None

    def generate(self, prompt: str, **kwargs: Any) -> List[GeneratedImage]:
        payload = json.dumps({
            "prompt": prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            resp_body = exc.read().decode(errors="replace")
            try:
                detail = json.loads(resp_body)
            except json.JSONDecodeError:
                detail = {"detail": resp_body}
            if exc.code == 422 and detail.get("error") == "guardrail":
                raise GuardrailError(self.name, detail.get("detail", "")) from exc
            raise RuntimeError(
                f"Flux2 server error {exc.code}: {detail}"
            ) from exc

        normalized: List[GeneratedImage] = []
        for img in body.get("images", []):
            b64_data: str | None = img.get("base64")
            local_path: str | None = None
            file_url: str | None = None
            vscode_url: str | None = None

            if b64_data and self.save_images:
                raw_bytes = base64.b64decode(b64_data)
                local_path, file_url, vscode_url = self._persist_image_bytes(raw_bytes)

            record = GeneratedImage(
                url=local_path or "",
                data_base64=b64_data,
                width=img.get("width", 0),
                height=img.get("height", 0),
                content_type=img.get("content_type", "image/png"),
                provider=self.name,
                extra={
                    "local_path": local_path,
                    "file_url": file_url,
                    "vscode_url": vscode_url,
                },
            )
            normalized.append(record)

        return normalized
