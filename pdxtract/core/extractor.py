"""Deterministic regex extractors."""

from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

# RFC-ish email regex. Avoids trailing punctuation and allows common characters.
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+'\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Phone regex: supports US/international-ish formats.
PHONE_REGEX = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

# URL regex: simple http/https/ftp capture.
URL_REGEX = re.compile(
    r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.~%-]*)?(?:\?(?:[\w&=%.\-]*))?)?"
)


def _clean_email(match: str) -> str | None:
    """Strip common trailing punctuation from emails."""
    email = match.rstrip(".,;:!?'")
    if ".." in email or email.count("@") != 1:
        return None
    local, domain = email.rsplit("@", 1)
    if not local or not domain:
        return None
    if "." not in domain:
        return None
    # Reject obviously invalid TLD lengths
    tld = domain.rsplit(".", 1)[-1]
    if len(tld) < 2:
        return None
    return email.lower()


def _clean_phone(match: str) -> str | None:
    digits = re.sub(r"\D", "", match)
    if len(digits) == 10:
        return digits
    if len(digits) == 11 and digits.startswith("1"):
        return digits[1:]
    return None


def _clean_url(match: str) -> str | None:
    url = match.rstrip(".,;:!?'")
    if len(url) < 4:
        return None
    return url


class ExtractorRegistry:
    """Registry of named extractors."""

    def __init__(self) -> None:
        self._extractors: dict[str, Callable[[str], list[str]]] = {
            "email": self.extract_emails,
            "phone": self.extract_phones,
            "url": self.extract_urls,
        }
        self._custom: dict[str, re.Pattern] = {}

    def register_custom(self, name: str, pattern: str) -> None:
        """Register a custom named regex extractor."""
        try:
            self._custom[name] = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex for extractor '{name}': {exc}") from exc

    def extract(self, name: str, text: str) -> list[str]:
        """Run extractor by name and return deduplicated matches in order."""
        if name in self._custom:
            return self._run_regex(self._custom[name], text, lambda m: m.group(0))
        if name in self._extractors:
            return self._extractors[name](text)
        raise ValueError(f"Unknown extractor '{name}'")

    def extract_many(self, names: list[str], text: str) -> dict[str, list[str]]:
        """Run multiple extractors and return a mapping name -> matches."""
        return {name: self.extract(name, text) for name in names}

    @staticmethod
    def _run_regex(
        pattern: re.Pattern,
        text: str,
        cleaner: Callable[[re.Match], str | None],
    ) -> list[str]:
        seen = set()
        results = []
        for match in pattern.finditer(text):
            cleaned = cleaner(match)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                results.append(cleaned)
        return results

    @classmethod
    def extract_emails(cls, text: str) -> list[str]:
        return cls._run_regex(EMAIL_REGEX, text, lambda m: _clean_email(m.group(0)))

    @classmethod
    def extract_phones(cls, text: str) -> list[str]:
        return cls._run_regex(PHONE_REGEX, text, lambda m: _clean_phone(m.group(0)))

    @classmethod
    def extract_urls(cls, text: str) -> list[str]:
        return cls._run_regex(URL_REGEX, text, lambda m: _clean_url(m.group(0)))


def build_extractor_registry(custom_regex: dict[str, str]) -> ExtractorRegistry:
    """Build an extractor registry with built-ins plus custom patterns."""
    registry = ExtractorRegistry()
    for name, pattern in (custom_regex or {}).items():
        registry.register_custom(name, pattern)
    return registry
