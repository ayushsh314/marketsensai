"""Live news: Finnhub company news (up to a year back on the free tier) plus yfinance's latest headlines."""

import hashlib
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from .data import ARTICLE_COLUMNS, extract_milestone_hint

FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
FINNHUB_MIN_INTERVAL = 1.05  # free tier allows 60 calls/minute
_NAME_SUFFIXES = r"\b(inc|corp|corporation|co|company|ltd|plc|holdings|group|platforms|\.com)\b\.?"


def _article_id(source: str, key: str) -> str:
    return f"{source}:{hashlib.sha1(key.encode()).hexdigest()[:16]}"


def _to_date(ts: datetime) -> datetime:
    return datetime(ts.year, ts.month, ts.day)


def _article(ticker, source, key, title, summary, published: datetime, publisher, url) -> Dict:
    title = (title or "").strip()
    summary = (summary or "").strip()
    day = _to_date(published)
    return {
        "article_id": _article_id(source, key),
        "ticker": ticker,
        "title": title,
        "content_clean": f"{title}. {summary}" if summary else title,
        "date": day.strftime("%Y-%m-%d"),
        "parsed_date": day,
        "published_at": published.astimezone(timezone.utc).isoformat(),
        "source": source,
        "publisher": publisher or "",
        "url": url or "",
        "milestone_hint": extract_milestone_hint(title),
    }


# ── Finnhub ──
def parse_finnhub_item(ticker: str, item: Dict) -> Optional[Dict]:
    if not item.get("headline") or not item.get("datetime"):
        return None
    published = datetime.fromtimestamp(item["datetime"], tz=timezone.utc)
    return _article(
        ticker, "finnhub", str(item.get("id") or item.get("url") or item["headline"]),
        item["headline"], item.get("summary"), published, item.get("source"), item.get("url"),
    )


def _finnhub_get(session, params: Dict, retries: int = 5) -> List[Dict]:
    for attempt in range(retries):
        resp = session.get(FINNHUB_URL, params=params, timeout=30)
        if resp.status_code == 429:  # rate limited — back off and retry
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in (401, 403):
            raise SystemExit("Finnhub rejected the API key. Set FINNHUB_API_KEY (free key at finnhub.io).")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    raise RuntimeError(f"Finnhub kept rate-limiting {params['symbol']} {params['from']}..{params['to']}")


def fetch_finnhub_news(ticker: str, start: datetime, end: datetime, api_key: str,
                       window_days: int = 7, session=None, throttle: float = FINNHUB_MIN_INTERVAL) -> List[Dict]:
    """Company news between start and end, requested window by window (Finnhub caps each response)."""
    import requests

    session = session or requests.Session()
    articles = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=window_days - 1), end)
        params = {"symbol": ticker, "from": cursor.strftime("%Y-%m-%d"),
                  "to": stop.strftime("%Y-%m-%d"), "token": api_key}
        for item in _finnhub_get(session, params):
            parsed = parse_finnhub_item(ticker, item)
            if parsed:
                articles.append(parsed)
        cursor = stop + timedelta(days=1)
        time.sleep(throttle)
    return articles


# ── yfinance ──
# Names the press uses that differ from the company's legal short name.
KEYWORD_ALIASES = {"GOOGL": ["google"], "GOOG": ["google"], "META": ["facebook"]}


def company_name(ticker: str) -> str:
    """Display name for prompts ("Apple Inc."); falls back to the ticker."""
    import yfinance as yf

    try:
        return yf.Ticker(ticker).info.get("shortName") or ticker
    except Exception:
        return ticker


def company_keywords(ticker: str) -> List[str]:
    """Words that identify the company in text: first word of its short name ("Apple Inc." → "apple") plus aliases."""
    import yfinance as yf

    keywords = list(KEYWORD_ALIASES.get(ticker, []))
    try:
        name = yf.Ticker(ticker).info.get("shortName") or ""
    except Exception:
        return keywords
    words = re.findall(r"[a-z]{3,}", re.sub(_NAME_SUFFIXES, " ", name.lower()))
    return words[:1] + keywords


def parse_yfinance_item(ticker: str, item: Dict) -> Optional[Dict]:
    if not isinstance(item, dict):
        return None
    content = item.get("content") or item  # yfinance ≥0.2.50 nests everything under "content" (sometimes null)
    if not isinstance(content, dict):
        return None
    title = content.get("title")
    if not title:
        return None
    if content.get("pubDate"):
        published = datetime.fromisoformat(content["pubDate"].replace("Z", "+00:00"))
    elif content.get("providerPublishTime"):
        published = datetime.fromtimestamp(content["providerPublishTime"], tz=timezone.utc)
    else:
        return None
    url = (content.get("canonicalUrl") or {}).get("url") or content.get("link")
    publisher = (content.get("provider") or {}).get("displayName") or content.get("publisher")
    return _article(ticker, "yfinance", str(item.get("id") or url or title), title,
                    content.get("summary"), published, publisher, url)


def is_relevant(article: Dict, ticker: str, keywords: List[str]) -> bool:
    text = f"{article['title']} {article['content_clean']}"
    if re.search(rf"\b{re.escape(ticker)}\b", text):
        return True
    return any(re.search(rf"\b{re.escape(k)}\b", text.lower()) for k in keywords)


def fetch_yfinance_news(ticker: str, count: int = 100) -> List[Dict]:
    """yfinance's latest headlines (roughly the last day or two). A supplement to Finnhub, so failures only warn."""
    import yfinance as yf

    try:
        items = yf.Ticker(ticker).get_news(count=count) or []
        articles = [parse_yfinance_item(ticker, item) for item in items]
    except Exception as e:
        print(f"   ⚠️ yfinance news failed for {ticker} ({type(e).__name__}: {e}); continuing with Finnhub only")
        return []
    return [a for a in articles if a]


# ── Combined ──
def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeats of the same story (same URL, or same headline for the same ticker)."""
    if df.empty:
        return df
    title_key = df["title"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    df = df[~(df["url"].ne("") & df.duplicated(subset=["ticker", "url"]))]
    return df[~pd.concat([df["ticker"], title_key.loc[df.index]], axis=1).duplicated()]


def fetch_live_news(tickers: List[str], start: datetime, end: datetime, api_key: Optional[str],
                    window_days: int = 7) -> pd.DataFrame:
    if not api_key:
        raise SystemExit("FINNHUB_API_KEY is not set. Get a free key at finnhub.io "
                         "(on Colab, add it under Secrets).")
    rows = []
    for ticker in tickers:
        # Both sources tag market-wide stories with many tickers; keep only ones that name the company.
        keywords = company_keywords(ticker)
        finnhub = fetch_finnhub_news(ticker, start, end, api_key, window_days=window_days)
        latest = [a for a in fetch_yfinance_news(ticker) if start <= a["parsed_date"] <= end]
        relevant = [a for a in finnhub + latest if is_relevant(a, ticker, keywords)]
        print(f"   📰 {ticker}: {len(finnhub)} Finnhub + {len(latest)} yfinance articles, "
              f"{len(relevant)} mention {'/'.join(keywords + [ticker])}")
        rows.extend(relevant)
    df = pd.DataFrame(rows, columns=ARTICLE_COLUMNS)
    before = len(df)
    df = dedupe(df).reset_index(drop=True)
    print(f"   Kept {len(df)} articles after removing {before - len(df)} duplicates")
    return df
