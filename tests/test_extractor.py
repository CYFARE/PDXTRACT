"""Tests for deterministic regex extractors."""

import pytest

from pdxtract.core.extractor import ExtractorRegistry, build_extractor_registry


def test_email_extraction():
    text = "Contact alice@example.com or bob.smith+tag@company.co.uk."
    registry = ExtractorRegistry()
    emails = registry.extract("email", text)
    assert emails == ["alice@example.com", "bob.smith+tag@company.co.uk"]


def test_email_deduplication():
    text = "Repeat repeat@example.com and repeat@example.com"
    registry = ExtractorRegistry()
    assert registry.extract("email", text) == ["repeat@example.com"]


def test_phone_extraction():
    text = "Call 555-123-4567 or (555) 987 6543."
    registry = ExtractorRegistry()
    phones = registry.extract("phone", text)
    assert "5551234567" in phones
    assert "5559876543" in phones


def test_url_extraction():
    text = "Visit https://example.com/path?x=1 or http://test.org."
    registry = ExtractorRegistry()
    urls = registry.extract("url", text)
    assert "https://example.com/path?x=1" in urls
    assert "http://test.org" in urls


def test_custom_regex():
    registry = build_extractor_registry({"case": r"Case\s+#\d+"})
    matches = registry.extract("case", "Case #123 and Case #456")
    assert matches == ["Case #123", "Case #456"]


def test_unknown_extractor_raises():
    registry = ExtractorRegistry()
    with pytest.raises(ValueError):
        registry.extract("does_not_exist", "text")
