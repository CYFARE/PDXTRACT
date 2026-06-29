"""Logging setup utilities."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO, use_rich: bool = True) -> None:
    """Configure root logger."""
    handlers: list[logging.Handler] = []

    if use_rich:
        try:
            from rich.logging import RichHandler

            handlers.append(
                RichHandler(
                    rich_tracebacks=True,
                    markup=False,
                    show_path=False,
                )
            )
        except Exception:
            use_rich = False

    if not handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        handlers.append(handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
