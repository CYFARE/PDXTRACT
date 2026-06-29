"""JSON/JSONL output writers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pdxtract.models import PdfResult

logger = logging.getLogger(__name__)


def append_jsonl(result: PdfResult, temp_file: str | Path) -> bool:
    """Append a single PDF result to the incremental JSONL file."""
    path = Path(temp_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(include_ocr_text=True)) + "\n")
        return True
    except OSError as exc:
        logger.error("Failed to append result to %s: %s", path, exc)
        return False


def read_jsonl(temp_file: str | Path) -> list[dict[str, Any]]:
    """Read all valid records from a JSONL file."""
    results: list[dict[str, Any]] = []
    path = Path(temp_file)
    if not path.exists():
        return results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSONL line in %s: %s", path, exc)
    return results


def finalize_json(
    temp_file: str | Path,
    output_file: str | Path,
    include_ocr_text: bool = False,
) -> bool:
    """Read temp JSONL, clean, and write final JSON output."""
    records = read_jsonl(temp_file)
    cleaned: list[dict[str, Any]] = []

    for record in records:
        if record.get("status") == "failed":
            continue
        pages = record.get("pages", [])
        cleaned_pages = []
        for page in pages:
            if page.get("status") != "success":
                continue
            extractions = page.get("extractions", {})
            if not any(extractions.values()):
                continue
            cleaned_pages.append(
                {
                    "page": page["page"],
                    "extractions": extractions,
                    **(
                        {"ocr_text": page["ocr_text"]}
                        if include_ocr_text and page.get("ocr_text")
                        else {}
                    ),
                    **(
                        {"used_embedded_text": page.get("used_embedded_text", False)}
                        if page.get("used_embedded_text")
                        else {}
                    ),
                }
            )
        if cleaned_pages:
            cleaned.append(
                {
                    "pdf_filename": record["pdf_filename"],
                    "pages": cleaned_pages,
                }
            )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        logger.info("Wrote final JSON to %s (%s PDFs with matches)", output_path, len(cleaned))
        return True
    except OSError as exc:
        logger.error("Failed to write final JSON to %s: %s", output_path, exc)
        return False


def remove_temp_file(temp_file: str | Path) -> None:
    path = Path(temp_file)
    if path.exists():
        try:
            os.remove(path)
            logger.info("Removed temp file %s", path)
        except OSError as exc:
            logger.warning("Could not remove temp file %s: %s", path, exc)
