"""Tests for the sponsor matcher: live list, SEED fallback, TTL cache, and the
jev-<7char> placeholder (no real network - the fetcher is monkeypatched)."""

import re

import pytest
from app.config import settings
from app.services import sponsors

# Names that are NOT in SEED_SPONSORS, so a match proves the live list drove it.
MOCK_LIST = {
    "sponsors": [
        {"name": "Zorptech", "aliases": ["Zorp Tech", "ZorpTechnologies"]},
        {"name": "Quibbly", "aliases": ["Quib"]},
    ]
}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """Fresh cache and default sponsor settings around every test."""
    sponsors.reset_cache()
    monkeypatch.setattr(settings, "MINUSPOD_SPONSORS_URL", None)
    monkeypatch.setattr(settings, "MINUSPOD_API_TOKEN", None)
    monkeypatch.setattr(settings, "SPONSOR_CACHE_TTL_SECONDS", 3600.0)
    yield
    sponsors.reset_cache()


def _fake_fetch(payload=MOCK_LIST, counter=None):
    def fetch(url, *, token, timeout):
        if counter is not None:
            counter.append(url)
        return payload

    return fetch


def test_matched_sponsor_name_and_alias(monkeypatch):
    monkeypatch.setattr(settings, "MINUSPOD_SPONSORS_URL", "https://mp.test/api/sponsors")
    monkeypatch.setattr(sponsors, "fetch_sponsors_json", _fake_fetch())
    assert sponsors.match_sponsor("try Zorptech today") == "Zorptech"  # name
    assert sponsors.match_sponsor("use Zorp Tech now") == "Zorptech"  # alias -> canonical
    assert sponsors.sponsor_for_span("grab some Quib later") == "Quibbly"  # alias via span


def test_unmatched_span_gets_jev_placeholder(monkeypatch):
    monkeypatch.setattr(settings, "MINUSPOD_SPONSORS_URL", "https://mp.test/api/sponsors")
    monkeypatch.setattr(sponsors, "fetch_sponsors_json", _fake_fetch())
    val = sponsors.sponsor_for_span("just a normal chat about the weather")
    assert re.fullmatch(r"jev-[a-z0-9]{7}", val)
    assert val != sponsors.sponsor_for_span("another unmatched span entirely")  # fresh per span


def test_fallback_to_seed_on_fetch_error(monkeypatch):
    monkeypatch.setattr(settings, "MINUSPOD_SPONSORS_URL", "https://mp.test/api/sponsors")

    def boom(url, *, token, timeout):
        raise RuntimeError("network down")

    monkeypatch.setattr(sponsors, "fetch_sponsors_json", boom)
    assert sponsors.match_sponsor("go to BetterHelp for therapy") == "BetterHelp"  # seed name


def test_fallback_to_seed_when_url_unset(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(sponsors, "fetch_sponsors_json", _fake_fetch(counter=calls))
    assert sponsors.match_sponsor("brought to you by Squarespace") == "Squarespace"
    assert calls == []  # URL unset -> SEED path, fetcher never called


def test_ttl_cache_fetches_once(monkeypatch):
    monkeypatch.setattr(settings, "MINUSPOD_SPONSORS_URL", "https://mp.test/api/sponsors")
    calls: list[str] = []
    monkeypatch.setattr(sponsors, "fetch_sponsors_json", _fake_fetch(counter=calls))
    assert sponsors.match_sponsor("try Zorptech today") == "Zorptech"
    assert sponsors.match_sponsor("grab some Quib later") == "Quibbly"
    assert len(calls) == 1  # second match served from the TTL cache
