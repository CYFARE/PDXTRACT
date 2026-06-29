"""Native GOT-OCR 2.0 provider via Hugging Face Transformers."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from pdxtract.providers.base import BaseOCRProvider

logger = logging.getLogger(__name__)


def _has_flash_attention() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


class GotOcr20Provider(BaseOCRProvider):
    """Native GOT-OCR 2.0 inference. Optional; requires transformers + torch."""

    name = "got_ocr20_native"
    is_thread_safe = False

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._model_id = config.model
        self._device_setting = config.got_ocr_device
        self._dtype_setting = config.got_ocr_dtype
        self._tokenizer = None
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Native GOT-OCR 2.0 requires torch and transformers. "
                "Install: pip install -r requirements-got.txt"
            ) from exc

        logger.info("Loading native GOT-OCR 2.0 model: %s", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )

        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, device)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "pad_token_id": self._tokenizer.eos_token_id,
            "torch_dtype": dtype,
            "device_map": "auto",
        }

        # Prefer flash-attention when available; otherwise let transformers pick
        # its default attention implementation (no flash-attn build required).
        if _has_flash_attention():
            load_kwargs["attn_implementation"] = "flash_attention_2"
            try:
                self._model = AutoModel.from_pretrained(self._model_id, **load_kwargs)
                logger.info("GOT-OCR 2.0 loaded with flash_attention_2")
                return
            except Exception as exc:
                logger.warning("flash_attention_2 failed (%s), retrying without it.", exc)
                load_kwargs.pop("attn_implementation", None)

        self._model = AutoModel.from_pretrained(self._model_id, **load_kwargs)
        logger.info("GOT-OCR 2.0 loaded successfully")

    def _resolve_device(self, torch: Any) -> str:
        if self._device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._device_setting

    def _resolve_dtype(self, torch: Any, device: str):
        if self._dtype_setting == "auto":
            if device == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            if device == "cuda":
                return torch.float16
            return torch.float32
        mapping = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        return mapping[self._dtype_setting]

    def health_check(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def ocr_image(self, image_bytes: bytes, page_num: int, pdf_path: str) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("GOT-OCR 2.0 model is not loaded.")

        pdf_filename = pdf_path.split("/")[-1].split("\\")[-1]

        # model.chat() in the reference implementation expects a file path.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            logger.debug("GOT-OCR 2.0 processing page %s of %s", page_num + 1, pdf_filename)
            result = self._model.chat(self._tokenizer, tmp_path, ocr_type="ocr")
            if not isinstance(result, str):
                result = str(result)
            logger.info("GOT-OCR 2.0 OK page %s of %s", page_num + 1, pdf_filename)
            return result
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
