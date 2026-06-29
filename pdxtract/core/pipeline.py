"""Main PDF processing pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

from pdxtract.core.extractor import ExtractorRegistry
from pdxtract.core.renderer import (
    PdfPageIterator,
    extract_embedded_text,
    render_page_to_bytes,
)
from pdxtract.models import PageResult, PdfResult
from pdxtract.providers.base import BaseOCRProvider

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Orchestrates render → OCR → extract for one PDF."""

    def __init__(
        self,
        provider: BaseOCRProvider,
        extractor_registry: ExtractorRegistry,
        config: Any,
    ) -> None:
        self.provider = provider
        self.extractors = extractor_registry
        self.config = config

    def _extract_from_text(self, text: str) -> dict[str, list[str]]:
        """Run all configured extractors on text."""
        return self.extractors.extract_many(self.config.extractors, text)

    def _has_any_match(self, extractions: dict[str, list[str]]) -> bool:
        return any(matches for matches in extractions.values())

    def process_pdf(self, pdf_path: str) -> PdfResult:
        pdf_filename = pdf_path.split("/")[-1].split("\\")[-1]
        pdf_start = time.time()
        result = PdfResult(pdf_filename=pdf_filename)

        try:
            with PdfPageIterator(pdf_path) as pages:
                num_pages = len(pages)
                logger.info("Processing %s (%s pages)", pdf_filename, num_pages)

                for page_idx, page in pages:
                    page_start = time.time()
                    page_result = PageResult(page=page_idx + 1)

                    try:
                        text_source: str | None = None
                        used_embedded = False

                        # 1. Hybrid shortcut: try embedded text first
                        if self.config.strategy == "hybrid" and self.config.use_embedded_text:
                            embedded_text = extract_embedded_text(page)
                            if embedded_text.strip():
                                extractions = self._extract_from_text(embedded_text)
                                if self._has_any_match(extractions):
                                    text_source = embedded_text
                                    used_embedded = True
                                    page_result.status = "success"
                                    page_result.used_embedded_text = True
                                    page_result.ocr_text = embedded_text
                                    page_result.extractions = extractions

                        # 2. OCR if no embedded match (or strategy forces OCR)
                        if text_source is None:
                            image_bytes = render_page_to_bytes(page, dpi=self.config.page_dpi)
                            if self.config.strategy == "vlm_regex":
                                # Provider may apply its own prompt; pass raw image.
                                ocr_text = self.provider.ocr_image(image_bytes, page_idx, pdf_path)
                            else:
                                ocr_text = self.provider.ocr_image(image_bytes, page_idx, pdf_path)
                            page_result.ocr_text = ocr_text
                            page_result.extractions = self._extract_from_text(ocr_text)
                            page_result.status = "success"

                    except Exception as exc:
                        logger.error("Error on page %s of %s: %s", page_idx + 1, pdf_filename, exc)
                        page_result.status = "failed"
                        page_result.error = str(exc)

                    page_result.elapsed_seconds = time.time() - page_start
                    result.pages.append(page_result)

            result.status = "success"
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf_filename, exc)
            result.status = "failed"
            result.error = str(exc)

        result.elapsed_seconds = time.time() - pdf_start
        return result
