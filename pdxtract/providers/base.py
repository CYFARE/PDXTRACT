"""Abstract OCR provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseOCRProvider(ABC):
    """All OCR/vision backends implement this interface."""

    name: str = "base"

    # Subclasses that are not thread-safe should override this.
    is_thread_safe: bool = True

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def ocr_image(self, image_bytes: bytes, page_num: int, pdf_path: str) -> str:
        """Return raw transcribed text for the given image."""
        ...

    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready."""
        return True

    def list_models(self) -> list[str] | None:
        """Return a list of available models, if the backend supports listing."""
        return None

    def close(self) -> None:
        """Release resources."""
        pass

    def __enter__(self) -> "BaseOCRProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
