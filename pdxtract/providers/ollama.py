"""Ollama OCR/Vision provider."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

from pdxtract.providers.base import BaseOCRProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseOCRProvider):
    """OCR provider that talks to a local Ollama server."""

    name = "ollama"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._client = None
        self._host = config.ollama_url
        self._model = config.model
        self._prompt = config.prompt
        self._retries = config.retries
        self._retry_delay = config.retry_delay_seconds
        self._ensure_client()

    def _ensure_client(self) -> None:
        try:
            import ollama
        except ImportError as exc:
            raise ImportError("The 'ollama' package is required for the Ollama provider.") from exc

        if self._client is None:
            self._client = ollama.Client(host=self._host)

    def health_check(self) -> bool:
        try:
            self._ensure_client()
            self._client.list()
            return True
        except Exception as exc:
            logger.error("Ollama health check failed at %s: %s", self._host, exc)
            return False

    def list_models(self) -> list[str] | None:
        try:
            self._ensure_client()
            response = self._client.list()
            models = response.get("models", [])
            return [m.get("model", m.get("name", "unknown")) for m in models]
        except Exception as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return None

    def ocr_image(self, image_bytes: bytes, page_num: int, pdf_path: str) -> str:
        pdf_filename = pdf_path.split("/")[-1].split("\\")[-1]
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self._retries:
            try:
                self._ensure_client()
                response = self._client.chat(
                    model=self._model,
                    messages=[
                        {
                            "role": "user",
                            "content": self._prompt,
                            "images": [image_bytes],
                        }
                    ],
                    options={"temperature": 0.0},
                )
                text = response["message"]["content"]
                logger.info("Ollama OK page %s of %s", page_num + 1, pdf_filename)
                return text
            except Exception as exc:
                err_msg = str(exc)
                if "not found" in err_msg.lower():
                    logger.error(
                        "Ollama model '%s' not found. Pull it with: ollama pull %s",
                        self._model,
                        self._model,
                    )
                    raise
                last_error = exc
                attempt += 1
                logger.warning(
                    "Ollama error on page %s of %s (attempt %s/%s): %s",
                    page_num + 1,
                    pdf_filename,
                    attempt,
                    self._retries + 1,
                    exc,
                )
                if attempt <= self._retries:
                    time.sleep(self._retry_delay)

        raise RuntimeError(
            f"Ollama failed page {page_num + 1} of {pdf_filename} after retries"
        ) from last_error

    def close(self) -> None:
        self._client = None
