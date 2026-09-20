"""Tests for the sponsor matcher: cookie login, session reuse, 401 re-login, SEED
fallback, TTL cache, and the jev-<7char> placeholder (no real network - the login
and fetch seams are monkeypatched)."""

import re
import time
from concurrent.futures import ThreadPoolExecutor

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
    """Fresh cache/session and default sponsor settings around every test."""
    sponsors.reset_cache()
    monkeypatch.setattr(settings, "MINUSPOD_BASE_URL", None)
    monkeypatch.setattr(settings, "MINUSPOD_PASSWORD", None)
    monkeypatch.setattr(settings, "SPONSOR_CACHE_TTL_SECONDS", 3600.0)
    monkeypatch.setattr(settings, "MINUSPOD_SESSION_TTL_SECONDS", 1800.0)
    yield
    sponsors.reset_cache()


def _configure(monkeypatch):
    monkeypatch.setattr(settings, "MINUSPOD_BASE_URL", "https://mp.test")
    monkeypatch.setattr(settings, "MINUSPOD_PASSWORD", "pw")


def _fake_login(counter=None):
    def _login(base_url, password, timeout):
        if counter is not None:
            counter.append(base_url)
        return {"session": "abc"}  # a fresh dict per login

    return _login


def _fake_fetch(payload=MOCK_LIST, counter=None):
    def _fetch(url, cookies, timeout):
        if counter is not None:
            counter.append(url)
        return payload

    return _fetch


def test_matched_sponsor_name_and_alias(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(sponsors, "login", _fake_login())
    monkeypatch.setattr(sponsors, "fetch_sponsors", _fake_fetch())
    assert sponsors.match_sponsor("try Zorptech today") == "Zorptech"  # name
    assert sponsors.match_sponsor("use Zorp Tech now") == "Zorptech"  # alias -> canonical
    assert sponsors.sponsor_for_span("grab some Quib later") == "Quibbly"  # alias via span


def test_authoritative_match_uses_aliases(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(sponsors, "login", _fake_login())
    monkeypatch.setattr(sponsors, "fetch_sponsors", _fake_fetch())
    assert sponsors.matched_sponsor_for_span("This episode is sponsored by Zorp Tech") == "Zorptech"


def test_authoritative_match_excludes_placeholder_rows(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(sponsors, "login", _fake_login())
    monkeypatch.setattr(
        sponsors,
        "fetch_sponsors",
        lambda url, cookies, timeout: {"sponsors": [{"name": "jev-old"}, {"name": "Zorptech"}]},
    )
    assert sponsors.matched_sponsor_for_span("This episode is sponsored by jev-old") is None


def test_refresh_timeout_does_not_return_rows_for_old_base_url(monkeypatch):
    sponsors._cache.update(rows=[("OldBrand", "OldBrand")], built_at=0.0,
                           url="https://old.test", failure_until=0.0)
    monkeypatch.setattr(settings, "MINUSPOD_BASE_URL", "https://new.test")
    sponsors._refresh_lock.acquire()
    try:
        assert sponsors.match_sponsor("OldBrand", deadline_at=sponsors.time.monotonic() + 0.001) is None
    finally:
        sponsors._refresh_lock.release()


def test_matched_sponsor_requires_promotional_evidence(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(sponsors, "login", _fake_login())
    monkeypatch.setattr(sponsors, "fetch_sponsors", _fake_fetch())
    assert sponsors.matched_sponsor_for_span("This episode is sponsored by Zorptech") == "Zorptech"
    assert sponsors.matched_sponsor_for_span("I have been using Zorptech for years") is None
    assert sponsors.matched_sponsor_for_span("Try Zorptech today with promo code SHOW") == "Zorptech"


def test_explicit_later_advertiser_beats_incidental_earlier_name(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(sponsors, "login", _fake_login())
    monkeypatch.setattr(sponsors, "fetch_sponsors", _fake_fetch(payload={"sponsors": [
        {"name": "Zorptech"}, {"name": "Simple"}, {"name": "Levels"},
        {"name": "Security Now"}, {"name": "Keeps"},
    ]}))
    text = "Simple ideas keep costs down. This episode is sponsored by Zorptech."
    assert sponsors.matched_sponsor_for_span(text) == "Zorptech"
    assert sponsors.matched_sponsor_for_span("Try simple ideas today") is None
    assert sponsors.matched_sponsor_for_span("Sponsored by Zorptech. Simple ideas are useful.") == "Zorptech"
    assert sponsors.matched_sponsor_for_span("This episode is not sponsored by Levels.") is None
    assert sponsors.matched_sponsor_for_span("Levels offer useful health advice.") is None


def test_multiple_promoted_brands_are_uncertain(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(sponsors, "login", _fake_login())
    monkeypatch.setattr(
        sponsors,
        "fetch_sponsors",
        lambda url, cookies, timeout: {
            "sponsors": [{"name": "Zorptech"}, {"name": "Quibbly"}]
        },
    )
    text = "Sponsored by Zorptech, with Quibbly promo code QUIB for a discount today."
    assert sponsors.matched_sponsor_for_span(text) is None


def test_unmatched_span_gets_jev_placeholder(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(sponsors, "login", _fake_login())
    monkeypatch.setattr(sponsors, "fetch_sponsors", _fake_fetch())
    val = sponsors.sponsor_for_span("just a normal chat about the weather")
    assert re.fullmatch(r"jev-[a-z0-9]{7}", val)
    assert val != sponsors.sponsor_for_span("another unmatched span entirely")  # fresh per span
    assert sponsors.matched_sponsor_for_span("just a normal chat about the weather") is None


def test_fallback_to_seed_when_unconfigured(monkeypatch):
    login_calls: list[str] = []
    fetch_calls: list[str] = []
    monkeypatch.setattr(sponsors, "login", _fake_login(counter=login_calls))
    monkeypatch.setattr(sponsors, "fetch_sponsors", _fake_fetch(counter=fetch_calls))
    assert sponsors.match_sponsor("brought to you by Squarespace") == "Squarespace"  # seed name
    assert login_calls == []  # unconfigured -> SEED path, never logs in
    assert fetch_calls == []
    assert sponsors.matched_sponsor_for_span("This episode is sponsored by AG1") == "Athletic Greens"


def test_fallback_to_seed_on_login_failure(monkeypatch):
    _configure(monkeypatch)
    login_calls: list[int] = []

    def boom(base_url, password, timeout):
        login_calls.append(1)
        raise RuntimeError("login rate-limited (HTTP 429)")

    monkeypatch.setattr(sponsors, "login", boom)
    assert sponsors.match_sponsor("go to BetterHelp for therapy") == "BetterHelp"  # seed name
    assert sponsors.match_sponsor("go to BetterHelp for therapy") == "BetterHelp"
    assert len(login_calls) == 1


def test_failed_refresh_is_single_flight_across_concurrent_matches(monkeypatch):
    _configure(monkeypatch)
    login_calls: list[int] = []

    def boom(base_url, password, timeout):
        login_calls.append(1)
        time.sleep(0.02)
        raise RuntimeError("rate limited")

    monkeypatch.setattr(sponsors, "login", boom)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: sponsors.match_sponsor("BetterHelp"), range(4)))
    assert results == ["BetterHelp"] * 4
    assert len(login_calls) == 1


def test_failed_refresh_retries_after_cooldown_expiry(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(sponsors, "login", _fake_login())
    monkeypatch.setattr(sponsors, "fetch_sponsors", _fake_fetch())
    sponsors._cache.update(rows=sponsors._rows_from_seed(), built_at=0.0,
                           url="https://mp.test", failure_until=100.0)
    monkeypatch.setattr(sponsors.time, "monotonic", lambda: 101.0)
    assert sponsors.match_sponsor("Zorptech") == "Zorptech"


def test_fallback_to_seed_on_fetch_error(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(sponsors, "login", _fake_login())

    def boom(url, cookies, timeout):
        raise RuntimeError("network down")

    monkeypatch.setattr(sponsors, "fetch_sponsors", boom)
    assert sponsors.match_sponsor("go to BetterHelp for therapy") == "BetterHelp"  # seed name


@pytest.mark.parametrize("payload", [{"sponsors": 1}, {"sponsors": [{"name": "x", "aliases": 1}]}])
def test_malformed_live_rows_fall_back_to_seed(monkeypatch, payload):
    _configure(monkeypatch)
    monkeypatch.setattr(sponsors, "login", _fake_login())
    monkeypatch.setattr(sponsors, "fetch_sponsors", _fake_fetch(payload=payload))
    assert sponsors.match_sponsor("go to BetterHelp for therapy") == "BetterHelp"


def test_session_reused_across_fetches(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(settings, "SPONSOR_CACHE_TTL_SECONDS", 0.0)  # force a refetch each match
    login_calls: list[str] = []
    fetch_calls: list[str] = []
    monkeypatch.setattr(sponsors, "login", _fake_login(counter=login_calls))
    monkeypatch.setattr(sponsors, "fetch_sponsors", _fake_fetch(counter=fetch_calls))
    assert sponsors.match_sponsor("try Zorptech today") == "Zorptech"
    assert sponsors.match_sponsor("grab some Quib later") == "Quibbly"
    assert len(fetch_calls) == 2  # matcher cache disabled -> two fetches
    assert len(login_calls) == 1  # session reused within TTL -> exactly one login


def test_401_triggers_single_relogin_then_succeeds(monkeypatch):
    _configure(monkeypatch)
    login_calls: list[str] = []
    monkeypatch.setattr(sponsors, "login", _fake_login(counter=login_calls))
    fetch_calls: list[str] = []

    def fetch(url, cookies, timeout):
        fetch_calls.append(url)
        if len(fetch_calls) == 1:
            raise sponsors.SponsorAuthError("401")
        return MOCK_LIST

    monkeypatch.setattr(sponsors, "fetch_sponsors", fetch)
    assert sponsors.match_sponsor("try Zorptech today") == "Zorptech"
    assert len(login_calls) == 2  # initial login + exactly one re-login after 401
    assert len(fetch_calls) == 2  # first GET 401s, the retry succeeds


def test_401_relogin_failure_falls_back_to_seed(monkeypatch):
    _configure(monkeypatch)
    login_calls: list[str] = []

    def login(base_url, password, timeout):
        login_calls.append(base_url)
        if len(login_calls) >= 2:
            raise RuntimeError("re-login rate-limited (HTTP 429)")
        return {"session": "abc"}

    def fetch(url, cookies, timeout):
        raise sponsors.SponsorAuthError("401")

    monkeypatch.setattr(sponsors, "login", login)
    monkeypatch.setattr(sponsors, "fetch_sponsors", fetch)
    assert sponsors.match_sponsor("go to BetterHelp for therapy") == "BetterHelp"  # seed name
    assert len(login_calls) == 2  # initial login + one re-login attempt, then SEED


def test_matcher_ttl_fetches_once(monkeypatch):
    _configure(monkeypatch)
    fetch_calls: list[str] = []
    monkeypatch.setattr(sponsors, "login", _fake_login())
    monkeypatch.setattr(sponsors, "fetch_sponsors", _fake_fetch(counter=fetch_calls))
    assert sponsors.match_sponsor("try Zorptech today") == "Zorptech"
    assert sponsors.match_sponsor("grab some Quib later") == "Quibbly"
    assert len(fetch_calls) == 1  # second match served from the TTL cache
