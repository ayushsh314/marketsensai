"""Stage 1 — data pipeline: build the article corpus (live or Kaggle) and select articles per time range."""

import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

ARTICLE_COLUMNS = [
    "article_id", "ticker", "title", "content_clean", "date", "parsed_date", "published_at",
    "source", "publisher", "url", "milestone_hint",
]
MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16

TIME_RANGE_DELTAS = {
    "1D": timedelta(days=1), "1W": timedelta(weeks=1), "1M": timedelta(days=30),
    "3M": timedelta(days=91), "6M": timedelta(days=182), "1Y": timedelta(days=365),
}


# ── Weak labels from headlines ──
def _horizon_from_weeks(weeks: int) -> Optional[str]:
    if 3 <= weeks <= 5:
        return "4-Week"
    if 50 <= weeks <= 53:
        return "52-Week"
    if 65 <= weeks <= 72:
        return "16-Month"
    return None  # e.g. a 13-week high has no taxonomy bucket


def _horizon_from_months(months: int) -> Optional[str]:
    return {1: "4-Week", 12: "52-Week", 16: "16-Month"}.get(months)


def extract_milestone_hint(title: str) -> Optional[str]:
    """Regex milestone label from a headline, used as a weak label for the hint-match metric."""
    tl = str(title).lower()
    direction = None
    for word, label in (("high", "High"), ("low", "Low")):
        if re.search(rf"\b{word}s?\b", tl):
            direction = label
            break
    if direction:
        if re.search(r"all[\s-]time|record", tl):
            return f"All-Time {direction}"
        m = re.search(r"(\d+)[\s-]week", tl)
        if m and _horizon_from_weeks(int(m.group(1))):
            return f"{_horizon_from_weeks(int(m.group(1)))} {direction}"
        m = re.search(r"(\d+)[\s-]month", tl)
        if m and _horizon_from_months(int(m.group(1))):
            return f"{_horizon_from_months(int(m.group(1)))} {direction}"
        if re.search(r"\b(1|one)[\s-]year\b", tl):
            return f"52-Week {direction}"
    if "breakout" in tl or "breaks out" in tl:
        return "Breakout Event"
    return None


# ── Kaggle source (the paper's static dataset) ──
KAGGLE_COMPANIES = {"apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN"}

# Relative dates ("18 days ago") in the dataset are anchored to its scrape date.
DATASET_REFERENCE_DATE = datetime(2023, 6, 15)


def parse_kaggle_row(row) -> pd.Series:
    """Parse the structured Content field of one Kaggle row."""
    title = str(row.get("Title", ""))
    content = str(row.get("Content", ""))
    tag = str(row.get("Tag", ""))

    # Every company named as a whole word (an article can cover several)
    text_lower = (title + " " + content).lower()
    tickers = [sym for name, sym in KAGGLE_COMPANIES.items() if re.search(rf"\b{name}\b", text_lower)]

    date_parsed = None
    iso_match = re.search(r'(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}', content)
    if iso_match:
        try:
            date_parsed = datetime.strptime(iso_match.group(1), "%Y-%m-%d")
        except ValueError:
            pass
    if date_parsed is None:
        rel_match = re.search(r'(\d+)\s*days?\s*ago', content)
        if rel_match:
            date_parsed = DATASET_REFERENCE_DATE - timedelta(days=int(rel_match.group(1)))

    # Price comes before a date like "186.612023-06-16", so stop before the year
    price = None
    price_match = re.search(r'(?:high|low)\s+of\s+(\d{1,6}\.\d{1,4})(?=\d{4}-|\d+\s*day|$|\s)', content)
    if price_match:
        price = float(price_match.group(1))

    clean = content
    if title and content.startswith(title):
        clean = content[len(title):]
    clean = re.sub(r'^United States\s*[^\w]*\s*', '', clean)
    if tag and tag.lower() in clean[:50].lower():
        idx = clean.lower().find(tag.lower())
        clean = clean[idx + len(tag):]
    clean = re.sub(r'https?://\S+|<[^>]+>', '', clean)
    clean = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\d.]*', '', clean)
    clean = re.sub(r'\d+\s*days?\s*ago', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    return pd.Series({
        "tickers": tickers,
        "title": title,
        "content_clean": clean if len(clean) > 20 else content,
        "date": date_parsed.strftime("%Y-%m-%d") if date_parsed else "unknown",
        "parsed_date": date_parsed,
        "price": price,
        "milestone_hint": extract_milestone_hint(title),
    })


def load_kaggle(csv_path: str, tickers: List[str]) -> pd.DataFrame:
    unsupported = set(tickers) - set(KAGGLE_COMPANIES.values())
    if unsupported:
        raise SystemExit(f"The Kaggle dataset only covers {sorted(KAGGLE_COMPANIES.values())}; "
                         f"got {sorted(unsupported)}. Use --source live for other tickers.")
    df_raw = pd.read_csv(csv_path, on_bad_lines="skip", encoding_errors="replace")
    parsed = df_raw.apply(parse_kaggle_row, axis=1)
    parsed["row"] = range(len(parsed))
    df = parsed.explode("tickers").rename(columns={"tickers": "ticker"})
    df = df[df["ticker"].isin(tickers) & (df["content_clean"].str.len() > 20)].copy()
    df["article_id"] = "kaggle:" + df["row"].astype(str)
    df["source"], df["publisher"], df["url"] = "kaggle", "", ""
    df["published_at"] = None  # the dataset only has dates
    print(f"   Rows: {len(df_raw):,} | Articles for {tickers}: {len(df):,} "
          f"(dated: {df['parsed_date'].notna().sum():,})")
    return df[ARTICLE_COLUMNS].reset_index(drop=True)


# ── Corpus ──
def build_corpus(cfg) -> Tuple[pd.DataFrame, datetime]:
    """Fetch (live) or load (kaggle) every article the longest time range needs; returns (corpus, as_of)."""
    lookback = max(TIME_RANGE_DELTAS[r] for r in cfg.time_ranges)
    if cfg.news_source == "kaggle":
        df = load_kaggle(cfg.data_path, cfg.tickers)
        as_of = cfg.as_of or df["parsed_date"].max().to_pydatetime()
    elif cfg.news_source == "live":
        from .news import fetch_live_news

        today = datetime.now()
        as_of = cfg.as_of or datetime(today.year, today.month, today.day)
        df = fetch_live_news(cfg.tickers, as_of - lookback, as_of, cfg.finnhub_api_key,
                             window_days=cfg.finnhub_window_days)
    else:
        raise ValueError(f"Unknown news source {cfg.news_source!r}")
    return df, as_of


def save_corpus(df: pd.DataFrame, as_of: datetime, path: str) -> None:
    out = df.assign(parsed_date=df["parsed_date"].dt.strftime("%Y-%m-%d")).astype(object)
    records = out.where(out.notna(), None).to_dict(orient="records")
    with open(path, "w") as f:
        json.dump({"as_of": as_of.strftime("%Y-%m-%d"), "articles": records}, f, indent=1)


def load_corpus(path: str) -> Tuple[pd.DataFrame, datetime]:
    with open(path) as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["articles"], columns=ARTICLE_COLUMNS)
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])
    return df, datetime.strptime(payload["as_of"], "%Y-%m-%d")


# ── Per-range selection ──
def range_bounds(time_range: str, as_of: datetime) -> Tuple[datetime, datetime]:
    return as_of - TIME_RANGE_DELTAS[time_range], as_of


def select_articles(df: pd.DataFrame, ticker: str, time_range: str, as_of: datetime,
                    max_articles: Optional[int] = None) -> pd.DataFrame:
    """Every article for ticker in (as_of − range, as_of], newest first.

    With max_articles set (for quick test runs), the range is split into that many equal time bins
    and articles are taken round-robin across bins, so the sample still spans the whole range.
    """
    start, end = range_bounds(time_range, as_of)
    d = df[(df["ticker"] == ticker) & df["parsed_date"].notna()]
    d = d[(d["parsed_date"] > start) & (d["parsed_date"] <= end)].sort_values("parsed_date")
    if max_articles and len(d) > max_articles:
        bins = pd.cut(d["parsed_date"], bins=max_articles, labels=False)
        rank = d.groupby(bins).cumcount()
        d = d.assign(_rank=rank.values, _bin=bins.values).sort_values(["_rank", "_bin"]).head(max_articles)
        d = d.drop(columns=["_rank", "_bin"])
    return d.sort_values("parsed_date", ascending=False)


def assign_trading_dates(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.Series:
    """The trading session each article can first affect.

    Published before the 4pm ET close → that day's session (or the next one, on weekends/holidays);
    published after the close → the next session. Articles with only a date map to that date's session.
    """
    days = pd.DatetimeIndex(sorted(trading_days))
    ts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    local = ts.dt.tz_convert(MARKET_TZ)
    day = local.dt.tz_localize(None).dt.normalize().fillna(pd.to_datetime(df["parsed_date"]).dt.normalize())
    after_close = (local.dt.hour >= MARKET_CLOSE_HOUR).fillna(False)
    day = day + pd.to_timedelta(after_close.astype(int), unit="D")
    idx = days.searchsorted(day.values, side="left")
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    ok = idx < len(days)
    out[ok] = days[idx[ok]]
    # Past the last known session (e.g. Friday after the close): the next weekday, not yet in the price data.
    late = ~ok & day.notna().values
    out[late] = (day[late] + pd.offsets.BDay(0)).values
    return out
