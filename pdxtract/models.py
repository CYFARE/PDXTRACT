"""Shared data models for PDXTRACT results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageResult:
    """Result for a single PDF page."""

    page: int
    status: str = "pending"  # pending, success, failed, skipped
    ocr_text: str | None = None
    extractions: dict[str, list[str]] = field(default_factory=dict)
    error: str | None = None
    elapsed_seconds: float = 0.0
    used_embedded_text: bool = False

    def to_dict(self, include_ocr_text: bool = True) -> dict[str, Any]:
        """Serialize to a dict for final output."""
        result: dict[str, Any] = {
            "page": self.page,
            "status": self.status,
            "extractions": self.extractions,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "used_embedded_text": self.used_embedded_text,
        }
        if include_ocr_text:
            result["ocr_text"] = self.ocr_text
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class PdfResult:
    """Result for a single PDF file."""

    pdf_filename: str
    status: str = "pending"
    pages: list[PageResult] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self, include_ocr_text: bool = True) -> dict[str, Any]:
        """Serialize to a dict for final output."""
        return {
            "pdf_filename": self.pdf_filename,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "pages": [p.to_dict(include_ocr_text=include_ocr_text) for p in self.pages],
            **({"error": self.error} if self.error else {}),
        }
