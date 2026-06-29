"""Resume session management."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_processed_files(session_file: str | Path) -> set[str]:
    """Load the set of previously successfully processed PDF filenames."""
    processed: set[str] = set()
    path = Path(session_file)
    if not path.exists():
        logger.info("No session file found at %s; starting fresh.", path)
        return processed
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    processed.add(name)
        logger.info("Loaded %s processed files from %s", len(processed), path)
    except OSError as exc:
        logger.error("Could not read session file %s: %s", path, exc)
    return processed


def save_processed_file(session_file: str | Path, filename: str) -> None:
    """Append a successfully processed filename to the session log."""
    try:
        with open(session_file, "a", encoding="utf-8") as f:
            f.write(filename + "\n")
    except OSError as exc:
        logger.error("Failed to write %s to session file %s: %s", filename, session_file, exc)


def is_processed(session_file: str | Path, filename: str) -> bool:
    return filename in load_processed_files(session_file)
