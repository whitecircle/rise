"""Provider that calls a local Qwen-Image inference server (scripts/qwen_image_server.py)."""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List, Optional
from uuid import uuid4

from image_red_teaming.providers.base import GeneratedImage, GuardrailError, ImageGenProvider

logger = logging.getLogger(__name__)


class QwenImageLocalProvider(ImageGenProvider):
    """HTTP client for the local Qwen-Image server.

    The server applies OpenAI input moderation and a Mistral prompt rewriter
    before generation; no output moderation is applied. A 422 with
    `error=guardrail` is converted into a `GuardrailError`.
    """

    name = "qwen-local"

    def __init__(
        self,
        port: int = 6000,
        host: str = "127.0.0.1",
        output_dir: str | Path | None = None,
        save_images: bool = True,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        width: int = 1024,
        height: int = 1024,
        negative_prompt: Optional[str] = None,
        timeout: float = 1800.0,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.output_dir = Path(output_dir) if output_dir else Path("logs/images")
        self.save_images = save_images
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.negative_prompt = negative_prompt
        self.timeout = timeout

    def _persist_image_bytes(
        self, image_bytes: bytes,
    ) -> tuple[str | None, str | None, str | None]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            filename = f"qwen_{uuid4().hex}.png"
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
        payload_data: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "width": self.width,
            "height": self.height,
        }
        if self.negative_prompt:
            payload_data["negative_prompt"] = self.negative_prompt
        payload = json.dumps(payload_data).encode()

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
                f"Qwen-Image server error {exc.code}: {detail}"
            ) from exc

        rewritten_prompt = body.get("rewritten_prompt")
        original_prompt = body.get("original_prompt", prompt)

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
                    "original_prompt": original_prompt,
                    "rewritten_prompt": rewritten_prompt,
                },
            )
            normalized.append(record)

        return normalized
