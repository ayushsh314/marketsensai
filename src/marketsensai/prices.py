"""Market data from yfinance, benchmark-adjusted returns, and price events detected from them.

Prices are the source of truth for *what happened*; the news pipeline explains *why*.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from .taxonomy import EVENT_TYPES, EXTREME_HORIZONS


@dataclass
class EventRules:
    vol_window: int = 60              # trailing trading days for volatility and beta
    move_z: float = 2.5               # |abnormal-return z| for Large Gain / Large Drop
    market_z: float = 2.5             # |raw-return z| for a market-wide move ...
    market_abn_z_max: float = 1.0     # ... that the benchmark explains
    gap_pct: float = 0.02             # open vs. previous close
    gap_z: float = 2.0
    volume_mult: float = 2.5          # vs. 50-day average volume
    breakout_lookback: int = 20
    breakout_volume_mult: float = 1.5
    min_history_all_time: int = 252   # ignore "all-time" extremes set in a stock's first year
    correction: float = 0.10          # drawdown from the 52-week high
    bear: float = 0.20


@dataclass
class MarketData:
    prices: Dict[str, pd.DataFrame]
    benchmark: pd.DataFrame
    features: Dict[str, pd.DataFrame] = field(default_factory=dict)
    events: Dict[str, pd.DataFrame] = field(default_factory=dict)     # one row per (day, label)
    event_days: Dict[str, pd.DataFrame] = field(default_factory=dict)  # one row per day

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        return self.benchmark.index


def fetch_price_history(ticker: str) -> pd.DataFrame:
    """Full daily OHLCV history (split-adjusted, not dividend-adjusted) indexed by naive date."""
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"yfinance returned no price history for {ticker}")
    hist.index = pd.DatetimeIndex(hist.index.date)
    return hist[["Open", "High", "Low", "Close", "Volume"]]


def compute_features(hist: pd.DataFrame, bench: pd.DataFrame, rules: EventRules) -> pd.DataFrame:
    """Daily returns, market-model abnormal returns (rolling beta vs. the benchmark), and trend state."""
    w = rules.vol_window
    close = hist["Close"]
    ret = close.pct_change()
    bench_ret = bench["Close"].reindex(hist.index).ffill().pct_change()
    beta = (ret.rolling(w).cov(bench_ret) / bench_ret.rolling(w).var()).shift(1)
    abn = ret - beta * bench_ret
    ret_std = ret.rolling(w).std().shift(1)
    abn_std = abn.rolling(w).std().shift(1)
    gap = hist["Open"] / close.shift(1) - 1
    return pd.DataFrame({
        "close": close,
        "ret": ret,
        "bench_ret": bench_ret,
        "beta": beta,
        "abn_ret": abn,
        "ret_z": ret / ret_std,
        "abn_z": abn / abn_std,
        "gap": gap,
        "gap_z": gap / ret_std,
        "vol_ratio": hist["Volume"] / hist["Volume"].rolling(50).mean().shift(1),
        "sma50": close.rolling(50).mean(),
        "sma200": close.rolling(200).mean(),
        "drawdown": close / hist["High"].rolling(252, min_periods=1).max() - 1,
    }, index=hist.index)


def _threshold_crossings(drawdown: pd.Series, level: float) -> pd.Series:
    """First close at least `level` below the high; re-arms only after recovering to within level/2."""
    hits = np.zeros(len(drawdown), dtype=bool)
    armed = True
    for i, dd in enumerate(drawdown.to_numpy()):
        if np.isnan(dd):
            continue
        if armed and dd <= -level:
            hits[i], armed = True, False
        elif not armed and dd > -level / 2:
            armed = True
    return pd.Series(hits, index=drawdown.index)


def detect_events(hist: pd.DataFrame, feats: pd.DataFrame, rules: EventRules) -> pd.DataFrame:
    """Every (day, event label) the price history satisfies; see taxonomy.EVENT_TYPES."""
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    days_of_history = pd.Series(np.arange(len(hist)), index=hist.index)
    flags = {}
    for horizon, n in EXTREME_HORIZONS.items():
        if n is None:
            prior_max, prior_min = high.cummax().shift(1), low.cummin().shift(1)
            enough = days_of_history >= rules.min_history_all_time
        else:
            prior_max, prior_min = high.rolling(n).max().shift(1), low.rolling(n).min().shift(1)
            enough = days_of_history >= n
        flags[f"{horizon} High"] = (high > prior_max) & enough
        flags[f"{horizon} Low"] = (low < prior_min) & enough

    abn_z, ret_z = feats["abn_z"], feats["ret_z"]
    flags["Large Gain"] = abn_z >= rules.move_z
    flags["Large Drop"] = abn_z <= -rules.move_z
    market_explained = abn_z.abs() < rules.market_abn_z_max
    flags["Market-Wide Rally"] = (ret_z >= rules.market_z) & market_explained
    flags["Market-Wide Selloff"] = (ret_z <= -rules.market_z) & market_explained
    flags["Gap Up"] = (feats["gap"] >= rules.gap_pct) & (feats["gap_z"] >= rules.gap_z)
    flags["Gap Down"] = (feats["gap"] <= -rules.gap_pct) & (feats["gap_z"] <= -rules.gap_z)
    flags["Volume Spike"] = feats["vol_ratio"] >= rules.volume_mult
    heavy = feats["vol_ratio"] >= rules.breakout_volume_mult
    flags["Breakout"] = (close > high.rolling(rules.breakout_lookback).max().shift(1)) & heavy
    flags["Breakdown"] = (close < low.rolling(rules.breakout_lookback).min().shift(1)) & heavy
    above = feats["sma50"] > feats["sma200"]
    valid = feats["sma200"].notna() & feats["sma200"].shift(1).notna()
    flags["Golden Cross"] = above & ~above.shift(1, fill_value=True) & valid
    flags["Death Cross"] = ~above & above.shift(1, fill_value=False) & valid
    flags["Correction"] = _threshold_crossings(feats["drawdown"], rules.correction)
    flags["Bear Market"] = _threshold_crossings(feats["drawdown"], rules.bear)

    flag_df = pd.DataFrame(flags).fillna(False).astype(bool)
    stacked = flag_df.stack()
    hits = stacked[stacked].reset_index()
    hits.columns = ["date", "label", "_"]
    if hits.empty:
        return pd.DataFrame(columns=["date", "label", "category", "direction", "weight", "ret", "abn_ret",
                                     "abn_z", "close"])
    meta = hits["label"].map(EVENT_TYPES)
    out = pd.DataFrame({
        "date": hits["date"],
        "label": hits["label"],
        "category": [m.category for m in meta],
        "direction": [m.direction for m in meta],
        "weight": [m.weight for m in meta],
    })
    return out.join(feats[["ret", "abn_ret", "abn_z", "close"]], on="date").reset_index(drop=True)


def summarize_event_days(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse events to one row per day with a significance score.

    On the high/low ladder only the strongest rung counts (a new all-time high is also a 4-week high,
    but that shouldn't add up), so significance = |abnormal z| + strongest extreme + other labels.
    """
    cols = ["date", "labels", "categories", "direction", "significance", "ret", "abn_ret", "abn_z", "close"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for date, g in events.groupby("date", sort=True):
        extremes, others = g[g["category"] == "extreme"], g[g["category"] != "extreme"]
        strongest = [extremes.loc[extremes[extremes["direction"] == d]["weight"].idxmax()]
                     for d in ("up", "down") if (extremes["direction"] == d).any()]
        kept = pd.concat([pd.DataFrame(strongest), others]) if strongest else others
        up = kept.loc[kept["direction"] == "up", "weight"].sum()
        down = kept.loc[kept["direction"] == "down", "weight"].sum()
        first = g.iloc[0]
        abn_z = first["abn_z"] if pd.notna(first["abn_z"]) else 0.0
        rows.append({
            "date": date,
            "labels": list(kept["label"]),
            "categories": sorted(set(kept["category"])),
            "direction": "up" if up > down else "down" if down > up else "neutral",
            "significance": round(float(abs(abn_z) + kept["weight"].sum()), 2),
            "ret": first["ret"], "abn_ret": first["abn_ret"], "abn_z": first["abn_z"], "close": first["close"],
        })
    return pd.DataFrame(rows, columns=cols)


def load_market(tickers: List[str], benchmark: str, rules: EventRules) -> MarketData:
    print(f"📈 Fetching price history from yfinance (benchmark {benchmark})...")
    bench = fetch_price_history(benchmark)
    market = MarketData(prices={}, benchmark=bench)
    for t in tickers:
        hist = fetch_price_history(t)
        feats = compute_features(hist, bench, rules)
        events = detect_events(hist, feats, rules)
        market.prices[t], market.features[t], market.events[t] = hist, feats, events
        market.event_days[t] = summarize_event_days(events)
        print(f"   📈 {t}: {len(hist):,} trading days ({hist.index[0].date()} → {hist.index[-1].date()}), "
              f"{len(market.event_days[t]):,} event days")
    return market


def between(df: pd.DataFrame, start: datetime, end: datetime, col: str = "date") -> pd.DataFrame:
    """Rows with start < df[col] <= end (or index, if col is None)."""
    if df.empty:
        return df
    key = df.index if col is None else df[col]
    return df[(key > pd.Timestamp(start)) & (key <= pd.Timestamp(end))]


def price_summary(hist: pd.DataFrame, bench: pd.DataFrame, start: datetime, end: datetime) -> Dict:
    window = between(hist, start, end, col=None)
    if window.empty:
        return {}
    b = between(bench, start, end, col=None)["Close"]
    ret = (window["Close"].iloc[-1] / window["Close"].iloc[0] - 1) * 100
    bench_ret = (b.iloc[-1] / b.iloc[0] - 1) * 100 if len(b) > 1 else float("nan")
    return {
        "start_date": str(window.index[0].date()),
        "end_date": str(window.index[-1].date()),
        "start_close": round(float(window["Close"].iloc[0]), 2),
        "end_close": round(float(window["Close"].iloc[-1]), 2),
        "return_pct": round(float(ret), 2),
        "benchmark_return_pct": round(float(bench_ret), 2),
        "excess_return_pct": round(float(ret - bench_ret), 2),
        "period_high": round(float(window["High"].max()), 2),
        "period_high_date": str(window["High"].idxmax().date()),
        "period_low": round(float(window["Low"].min()), 2),
        "period_low_date": str(window["Low"].idxmin().date()),
    }
