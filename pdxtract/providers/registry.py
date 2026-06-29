"""Provider registry/factory."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdxtract.config import Config
    from pdxtract.providers.base import BaseOCRProvider

logger = logging.getLogger(__name__)

_PROVIDER_CLASSES: dict[str, type["BaseOCRProvider"]] = {}


def _lazy_register() -> None:
    """Register providers lazily to avoid heavy imports at startup."""
    if _PROVIDER_CLASSES:
        return

    from pdxtract.providers.llama_cpp import LlamaCppProvider
    from pdxtract.providers.ollama import OllamaProvider

    _PROVIDER_CLASSES["ollama"] = OllamaProvider
    _PROVIDER_CLASSES["llama_cpp"] = LlamaCppProvider

    try:
        from pdxtract.providers.got_ocr20 import GotOcr20Provider

        _PROVIDER_CLASSES["got_ocr20_native"] = GotOcr20Provider
    except Exception as exc:
        logger.debug("Native GOT-OCR 2.0 provider not available: %s", exc)


def get_provider(config: "Config") -> "BaseOCRProvider":
    """Instantiate the configured provider."""
    _lazy_register()
    provider_name = config.provider
    cls = _PROVIDER_CLASSES.get(provider_name)
    if cls is None:
        available = ", ".join(_PROVIDER_CLASSES) or "none"
        raise ValueError(
            f"Unknown provider '{provider_name}'. Available: {available}. "
            f"If you want native GOT-OCR 2.0, install optional deps: pip install -r requirements-got.txt"
        )
    return cls(config)


def list_available_providers() -> list[str]:
    _lazy_register()
    return list(_PROVIDER_CLASSES.keys())
