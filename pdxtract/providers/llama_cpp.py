"""llama.cpp server OCR/Vision provider."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import requests

from pdxtract.providers.base import BaseOCRProvider

logger = logging.getLogger(__name__)


class LlamaCppProvider(BaseOCRProvider):
    """OCR provider that talks to a llama.cpp OpenAI-compatible server."""

    name = "llama_cpp"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._base_url = config.llama_cpp_url.rstrip("/")
        self._model = config.model
        self._prompt = config.prompt
        self._retries = config.retries
        self._retry_delay = config.retry_delay_seconds

    def _make_b64_image_url(self, image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self._base_url}/health", timeout=10)
            return resp.status_code == 200
        except Exception as exc:
            logger.error("llama.cpp health check failed at %s: %s", self._base_url, exc)
            return False

    def list_models(self) -> list[str] | None:
        try:
            resp = requests.get(f"{self._base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            return [m.get("id", "unknown") for m in models]
        except Exception as exc:
            logger.warning("Could not list llama.cpp models: %s", exc)
            return None

    def ocr_image(self, image_bytes: bytes, page_num: int, pdf_path: str) -> str:
        pdf_filename = pdf_path.split("/")[-1].split("\\")[-1]
        image_url = self._make_b64_image_url(image_bytes)
        attempt = 0
        last_error: Exception | None = None

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": self._prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 8192,
        }

        while attempt <= self._retries:
            try:
                resp = requests.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                logger.info("llama.cpp OK page %s of %s", page_num + 1, pdf_filename)
                return text
            except Exception as exc:
                last_error = exc
                attempt += 1
                logger.warning(
                    "llama.cpp error on page %s of %s (attempt %s/%s): %s",
                    page_num + 1,
                    pdf_filename,
                    attempt,
                    self._retries + 1,
                    exc,
                )
                if attempt <= self._retries:
                    time.sleep(self._retry_delay)

        raise RuntimeError(
            f"llama.cpp failed page {page_num + 1} of {pdf_filename} after retries"
        ) from last_error
