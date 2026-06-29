"""CSV and flat text output writers."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_csv(
    records: list[dict[str, Any]], output_file: str | Path, extractor_names: list[str]
) -> bool:
    """Write a flat CSV of all extraction matches."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for record in records:
        pdf = record.get("pdf_filename", "")
        for page in record.get("pages", []):
            page_num = str(page.get("page", ""))
            for extractor in extractor_names:
                for value in page.get("extractions", {}).get(extractor, []):
                    rows.append(
                        {
                            "pdf_filename": pdf,
                            "page": page_num,
                            "type": extractor,
                            "value": value,
                        }
                    )

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pdf_filename", "page", "type", "value"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote CSV to %s (%s rows)", output_path, len(rows))
        return True
    except OSError as exc:
        logger.error("Failed to write CSV to %s: %s", output_path, exc)
        return False


def write_txt(
    records: list[dict[str, Any]],
    output_file: str | Path,
    extractor_names: list[str],
) -> bool:
    """Write a deduplicated flat list of extraction values."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    values: list[str] = []
    for record in records:
        for page in record.get("pages", []):
            for extractor in extractor_names:
                for value in page.get("extractions", {}).get(extractor, []):
                    key = f"{extractor}:{value}"
                    if key not in seen:
                        seen.add(key)
                        values.append(value)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(values))
            if values:
                f.write("\n")
        logger.info("Wrote TXT to %s (%s unique values)", output_path, len(values))
        return True
    except OSError as exc:
        logger.error("Failed to write TXT to %s: %s", output_path, exc)
        return False
