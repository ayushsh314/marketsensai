from datetime import datetime

import pandas as pd

from marketsensai.data import ARTICLE_COLUMNS
from marketsensai.news import (
    dedupe,
    fetch_finnhub_news,
    is_relevant,
    parse_finnhub_item,
    parse_yfinance_item,
)


class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses, self.params = list(responses), []

    def get(self, url, params, timeout):
        self.params.append(params)
        return self.responses.pop(0)


FINNHUB_ITEM = {"id": 1, "datetime": 1758800000, "headline": "Apple hits 52-week high",
                "summary": "Shares rose.", "source": "Reuters", "url": "https://x/1", "related": "AAPL"}


def test_parse_finnhub_item():
    a = parse_finnhub_item("AAPL", FINNHUB_ITEM)
    assert a["ticker"] == "AAPL"
    assert a["content_clean"] == "Apple hits 52-week high. Shares rose."
    assert a["date"] == "2025-09-25"
    assert a["published_at"].startswith("2025-09-25T")
    assert a["milestone_hint"] == "52-Week High"
    assert set(a) == set(ARTICLE_COLUMNS)
    assert parse_finnhub_item("AAPL", {"headline": ""}) is None


def test_fetch_finnhub_news_windows_and_retries():
    session = FakeSession([
        FakeResponse([], status=429),       # rate limited, retried
        FakeResponse([FINNHUB_ITEM]),        # window 1
        FakeResponse([]),                    # window 2
    ])
    arts = fetch_finnhub_news("AAPL", datetime(2025, 9, 1), datetime(2025, 9, 10), "key",
                              window_days=7, session=session, throttle=0)
    assert len(arts) == 1
    assert [(p["from"], p["to"]) for p in session.params[1:]] == [("2025-09-01", "2025-09-07"),
                                                                  ("2025-09-08", "2025-09-10")]


def test_parse_yfinance_item_nested_format():
    item = {"id": "abc", "content": {
        "title": "Apple stock jumps", "summary": "Details.", "pubDate": "2026-09-25T19:38:05Z",
        "canonicalUrl": {"url": "https://y/1"}, "provider": {"displayName": "Yahoo Finance"}}}
    a = parse_yfinance_item("AAPL", item)
    assert a["date"] == "2026-09-25"
    assert a["url"] == "https://y/1"
    assert a["publisher"] == "Yahoo Finance"


def test_parse_yfinance_item_skips_malformed_items():
    assert parse_yfinance_item("AAPL", {"id": "x", "content": None}) is None
    assert parse_yfinance_item("AAPL", {"id": "x", "content": {"title": ""}}) is None
    assert parse_yfinance_item("AAPL", None) is None


def test_is_relevant():
    a = {"title": "Trump renames AI", "content_clean": "Trump renames AI. Nothing about the company."}
    assert not is_relevant(a, "AAPL", ["apple"])
    assert is_relevant({"title": "AAPL rallies", "content_clean": "AAPL rallies."}, "AAPL", ["apple"])
    assert is_relevant({"title": "Apple rallies", "content_clean": "Apple rallies."}, "AAPL", ["apple"])
    google = {"title": "Google unveils Gemini 4", "content_clean": "Google unveils Gemini 4."}
    assert is_relevant(google, "GOOGL", ["alphabet", "google"])


def test_dedupe_by_url_and_title():
    base = parse_finnhub_item("AAPL", FINNHUB_ITEM)
    same_url = {**base, "article_id": "other", "title": "Different headline"}
    same_title = {**base, "article_id": "third", "url": "https://x/other", "title": "Apple Hits 52-Week High!"}
    other_ticker = {**base, "ticker": "MSFT"}
    df = pd.DataFrame([base, same_url, same_title, other_ticker], columns=ARTICLE_COLUMNS)
    kept = dedupe(df)
    assert list(kept["ticker"]) == ["AAPL", "MSFT"]
