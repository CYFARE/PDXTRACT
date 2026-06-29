"""Configuration management for PDXTRACT."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_CONFIG_FILE = "config.json"


class Config(BaseModel):
    """Validated PDXTRACT configuration."""

    provider: str = Field(
        default="ollama", description="OCR backend: ollama, llama_cpp, got_ocr20_native"
    )
    model: str | None = Field(default=None, description="Model name or HuggingFace repo id")
    strategy: str = Field(
        default="ocr_regex", description="Extraction strategy: ocr_regex, hybrid, vlm_regex"
    )
    extractors: list[str] = Field(
        default_factory=lambda: ["email"], description="List of extractors to run"
    )
    custom_regex: dict[str, str] = Field(
        default_factory=dict, description="Name -> regex string map"
    )
    ollama_url: str = Field(default="http://127.0.0.1:11434", description="Ollama API base URL")
    llama_cpp_url: str = Field(
        default="http://127.0.0.1:8080", description="llama.cpp server base URL"
    )
    input_folder: str = Field(default="./pdfs", description="Folder containing input PDFs")
    output_file: str = Field(
        default="output/extracted_data.json", description="Final JSON output path"
    )
    output_csv: str | None = Field(default=None, description="Optional flat CSV output path")
    output_txt: str | None = Field(
        default=None, description="Optional deduplicated text list output path"
    )
    session_file: str = Field(default="aiocr_session.log", description="Resume session log")
    temp_suffix: str = Field(default=".jsonl.tmp", description="Suffix for incremental temp file")
    max_workers: int = Field(default=2, ge=1, description="Max concurrent PDFs/pages")
    page_dpi: int = Field(default=200, ge=72, le=600, description="DPI for page rendering")
    use_embedded_text: bool = Field(
        default=True, description="Try embedded text before OCR in hybrid mode"
    )
    include_ocr_text: bool = Field(default=False, description="Include raw OCR text in final JSON")
    prompt: str = Field(
        default="Transcribe all text visible in this image. Output only the raw text, no explanations.",
        description="Prompt sent to the vision/OCR model along with the image",
    )
    vlm_prompt: str | None = Field(
        default=None,
        description="Deprecated alias for prompt; copied to prompt if set",
    )
    got_ocr_device: str = Field(
        default="auto", description="Device for native GOT-OCR 2.0: auto, cuda, cpu"
    )
    got_ocr_dtype: str = Field(
        default="auto", description="Torch dtype for native GOT-OCR 2.0: auto, fp16, bf16, fp32"
    )
    retries: int = Field(default=2, ge=0, description="Retries per page on transient OCR failure")
    retry_delay_seconds: float = Field(default=3.0, ge=0)

    @field_validator("strategy")
    @classmethod
    def _valid_strategy(cls, v: str) -> str:
        allowed = {"ocr_regex", "hybrid", "vlm_regex"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}")
        return v

    @field_validator("provider")
    @classmethod
    def _valid_provider(cls, v: str) -> str:
        allowed = {"ollama", "llama_cpp", "got_ocr20_native"}
        if v not in allowed:
            raise ValueError(f"provider must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def _set_model_defaults(self) -> "Config":
        if self.model is None:
            defaults = {
                "ollama": "llama3.2-vision",
                "llama_cpp": "ggml-org/GLM-OCR-GGUF",
                "got_ocr20_native": "stepfun-ai/GOT-OCR2_0",
            }
            self.model = defaults.get(self.provider, "")
        if self.vlm_prompt:
            self.prompt = self.vlm_prompt
        return self

    def temp_output_file(self) -> str:
        """Return the temp JSONL file path based on the final output file."""
        return self.output_file + self.temp_suffix


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from JSON file, env vars, and defaults."""
    path = Path(path or os.environ.get("PDXTRACT_CONFIG", DEFAULT_CONFIG_FILE))
    data: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Simple env overrides
    env_map = {
        "PDXTRACT_PROVIDER": "provider",
        "PDXTRACT_MODEL": "model",
        "PDXTRACT_STRATEGY": "strategy",
        "PDXTRACT_OLLAMA_URL": "ollama_url",
        "PDXTRACT_LLAMA_CPP_URL": "llama_cpp_url",
        "PDXTRACT_INPUT_FOLDER": "input_folder",
        "PDXTRACT_OUTPUT_FILE": "output_file",
    }
    for env_key, cfg_key in env_map.items():
        value = os.environ.get(env_key)
        if value:
            data[cfg_key] = value

    return Config.model_validate(data)
