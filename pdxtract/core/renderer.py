"""PDF page rendering utilities."""

from __future__ import annotations

import io
import logging
from typing import Iterator

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


def render_page_to_bytes(page: fitz.Page, dpi: int = 200) -> bytes:
    """Render a PyMuPDF page to PNG bytes."""
    # Use a matrix for higher resolution
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    pix = None
    return img_bytes


def extract_embedded_text(page: fitz.Page) -> str:
    """Extract embedded text from a page using PyMuPDF."""
    text = page.get_text("text")
    return text or ""


class PdfPageIterator:
    """Iterate pages of a PDF as (page_num_0_indexed, fitz.Page) tuples."""

    def __init__(self, pdf_path: str) -> None:
        self.pdf_path = pdf_path
        self._doc: fitz.Document | None = None

    def __enter__(self) -> "PdfPageIterator":
        self._doc = fitz.open(self.pdf_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._doc:
            self._doc.close()
            self._doc = None

    def __iter__(self) -> Iterator[tuple[int, fitz.Page]]:
        if self._doc is None:
            raise RuntimeError("PdfPageIterator used outside context manager")
        for i in range(len(self._doc)):
            yield i, self._doc.load_page(i)

    def __len__(self) -> int:
        if self._doc is None:
            raise RuntimeError("PdfPageIterator used outside context manager")
        return len(self._doc)


def maybe_optimize_image(image_bytes: bytes, max_size: int = 4096) -> bytes:
    """Resize very large images while keeping aspect ratio; OCR models have limits."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        if width <= max_size and height <= max_size:
            return image_bytes
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Image optimization failed: %s", exc)
        return image_bytes
