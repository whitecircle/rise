from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypedDict, Optional


class GeneratedImage(TypedDict, total=False):
    """
    Minimal, provider-agnostic description of a generated image.

    Only `url` is required; providers can attach extra fields in `extra`.
    """

    url: str
    data_base64: Optional[str]
    width: int
    height: int
    content_type: str
    provider: str
    extra: Dict[str, Any]


class ImageGenProvider(ABC):
    """
    Simple interface for any image-generation provider.

    Your colleague should depend on this interface only, not concrete providers.
    """

    name: str

    @abstractmethod
    def generate(
        self, prompt: str, **kwargs: Any
    ) -> List[GeneratedImage]:
        """
        Generate one or more images from a text prompt.

        Returns a list of `GeneratedImage` dicts. At minimum, each dict has:
        - `url`: a public or fetchable URL for the image.
        """
        raise NotImplementedError


class GuardrailError(RuntimeError):
    """
    Raised when a provider blocks generation due to content guardrails.
    """

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(message)
        self.provider = provider
