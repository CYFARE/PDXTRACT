"""Tests for configuration loading and validation."""

import pydantic
import pytest

from pdxtract.config import Config


def test_default_config():
    cfg = Config()
    assert cfg.provider == "ollama"
    assert cfg.strategy == "ocr_regex"
    assert cfg.extractors == ["email"]


def test_model_default_per_provider():
    cfg = Config(provider="got_ocr20_native")
    assert cfg.model == "stepfun-ai/GOT-OCR2_0"

    cfg = Config(provider="llama_cpp")
    assert cfg.model == "ggml-org/GLM-OCR-GGUF"


def test_invalid_strategy_rejected():
    with pytest.raises(pydantic.ValidationError):
        Config(strategy="magic")


def test_invalid_provider_rejected():
    with pytest.raises(pydantic.ValidationError):
        Config(provider="openai")


def test_vlm_prompt_backward_compat():
    cfg = Config(vlm_prompt="Old prompt")
    assert cfg.prompt == "Old prompt"
