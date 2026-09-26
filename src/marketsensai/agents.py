"""The pipeline's agents.

    NewsRetrievalAgent      every article for a ticker in the longest time range
    ArticleAnalystAgent     LLM: relevance, sentiment, catalysts, price claims per article (cached on disk)
    StoryAgent              embeddings: groups articles covering the same news into stories
    EventAttributionAgent   LLM: explains each price-event day from the stories around it
    ReportAgent             LLM: one briefing per time range from all of the above
"""

import re
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cache import JsonlCache
from .config import Config
from .data import TIME_RANGE_DELTAS, range_bounds, select_articles
from .embeddings import embed_articles
from .llm import generate_json
from .prices import MarketData, between, price_summary
from .prompts import (
    ARTICLE_ANALYSIS_PROMPT,
    ARTICLE_ANALYSIS_SCHEMA,
    EVENT_ATTRIBUTION_PROMPT,
    EVENT_ATTRIBUTION_SCHEMA,
    REPORT_PROMPT,
)
from .stories import cluster_stories
from .taxonomy import CATALYST_CATEGORIES, catalyst_menu, normalize_catalyst, normalize_price_event

RELEVANT = 0.5  # articles at or above this relevance count toward sentiment and themes
ATTRIBUTION_DRIVERS = CATALYST_CATEGORIES + ["market-wide", "unexplained"]


def _to_float(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _clamp(value, lo: float, hi: float) -> Optional[float]:
    f = _to_float(value)
    return None if f is None else min(max(f, lo), hi)


def parse_price(value) -> Optional[float]:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(value or ""))
    return float(m.group(0).replace(",", "")) if m else None


# ── Agent 1: News retrieval ──
class NewsRetrievalAgent:
    """Every article for a ticker in the longest requested range (optionally capped for quick runs)."""

    def __init__(self, df: pd.DataFrame, as_of: datetime, time_ranges: List[str], max_articles: Optional[int]):
        self.df = df
        self.as_of = as_of
        self.longest = max(time_ranges, key=TIME_RANGE_DELTAS.get)
        self.max_articles = max_articles

    def retrieve(self, ticker: str) -> pd.DataFrame:
        selected = select_articles(self.df, ticker, self.longest, self.as_of, self.max_articles)
        print(f"   📰 NewsRetrieval: {len(selected):,} articles for {ticker} ({self.longest})")
        return selected


# ── Agent 2: Article analyst ──
def validate_analysis(parsed: dict) -> Dict:
    if not parsed:
        return {"parse_failed": True, "relevance": None, "sentiment": None, "catalysts": [], "price_claims": []}
    catalysts = []
    for c in parsed.get("catalysts") or []:
        if not isinstance(c, dict):
            continue
        raw = str(c.get("category") or "")
        label = str(c.get("label") or raw).strip()[:80]
        catalysts.append({
            "category": normalize_catalyst(raw, label),
            "label": label,
            "summary": str(c.get("summary") or "").strip()[:300],
        })
    claims = []
    for c in parsed.get("price_claims") or []:
        if isinstance(c, dict) and str(c.get("event") or "").strip():
            claims.append({"event": str(c["event"]).strip()[:80], "label": normalize_price_event(c["event"]),
                           "price": parse_price(c.get("price"))})
    relevance = _clamp(parsed.get("relevance"), 0.0, 1.0)
    return {
        "parse_failed": False,
        "relevance": 0.5 if relevance is None else relevance,
        "sentiment": _clamp(parsed.get("sentiment"), -1.0, 1.0),
        "catalysts": catalysts[:3],
        "price_claims": claims[:3],
    }


class ArticleAnalystAgent:
    """One structured LLM call per article; results are cached so each article is analyzed once, ever."""

    max_tokens = 400

    def __init__(self, llm, cfg: Config, company_names: Dict[str, str]):
        self.llm = llm
        self.cfg = cfg
        self.company_names = company_names
        self.cache = JsonlCache(f"{cfg.cache_dir}/article_analysis.jsonl" if cfg.cache_dir else None,
                                cfg.cache_namespace)

    def _prompt(self, ticker: str, row) -> str:
        published = row.get("published_at") or row["date"]
        return ARTICLE_ANALYSIS_PROMPT.format(
            company=self.company_names.get(ticker, ticker), ticker=ticker, published=published,
            article_text=str(row["content_clean"])[:self.cfg.max_article_chars], catalyst_menu=catalyst_menu(),
        )

    def analyze(self, ticker: str, articles: pd.DataFrame) -> Tuple[Dict[str, Dict], Dict]:
        t0 = time.time()
        keys = {aid: f"{ticker}|{aid}" for aid in articles["article_id"]}
        todo = articles[[keys[a] not in self.cache for a in articles["article_id"]]]
        batch = self.cfg.llm_batch_size
        for i in range(0, len(todo), batch):
            chunk = todo.iloc[i:i + batch]
            prompts = [self._prompt(ticker, row) for _, row in chunk.iterrows()]
            parsed = generate_json(self.llm, prompts, self.max_tokens, schema=ARTICLE_ANALYSIS_SCHEMA)
            self.cache.put_many((keys[aid], validate_analysis(p)) for aid, p in zip(chunk["article_id"], parsed))
            print(f"   🔎 {ticker}: analyzed {min(i + batch, len(todo)):,}/{len(todo):,} new articles")
        analyses = {aid: self.cache.get(k) for aid, k in keys.items()}
        stats = {
            "articles": len(articles),
            "articles_newly_analyzed": len(todo),
            "analysis_seconds": round(time.time() - t0, 1),
            "analysis_failures": sum(a["parse_failed"] for a in analyses.values()),
            "relevant_articles": sum((a["relevance"] or 0) >= RELEVANT for a in analyses.values()),
        }
        print(f"   ✅ {ticker}: {stats['relevant_articles']:,}/{len(articles):,} articles relevant, "
              f"{stats['analysis_failures']} unparseable, {len(todo):,} sent to the LLM")
        return analyses, stats


# ── Agent 3: Stories ──
class StoryAgent:
    """Clusters a ticker's articles into stories by embedding similarity and date."""

    def __init__(self, embed_model, cfg: Config):
        self.embed_model = embed_model
        self.cfg = cfg

    def cluster(self, articles: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        if articles.empty:
            return cluster_stories(articles, np.zeros((0, 1)), 1.0, 0)
        emb = embed_articles(articles, self.embed_model, self.cfg.embed_model_name, self.cfg.cache_dir)
        story_of, stories = cluster_stories(articles, emb, self.cfg.story_similarity, self.cfg.story_window_days)
        print(f"   🧵 {articles['ticker'].iloc[0]}: {len(articles):,} articles → {len(stories):,} stories")
        return story_of, stories


def profile_stories(stories: pd.DataFrame, articles: pd.DataFrame, analyses: Dict[str, Dict]) -> pd.DataFrame:
    """Add each story's relevance, relevance-weighted sentiment, catalysts, and a summary line."""
    if stories.empty:
        return stories.assign(relevance=[], sentiment=[], catalysts=[], summary=[])
    title_of = dict(zip(articles["article_id"], articles["title"]))
    rel, sent, cats, summaries = [], [], [], []
    for ids in stories["article_ids"]:
        a = [analyses[i] for i in ids if analyses.get(i) and not analyses[i]["parse_failed"]]
        weights = np.array([x["relevance"] for x in a]) if a else np.array([])
        scored = [(x["sentiment"], x["relevance"]) for x in a if x["sentiment"] is not None]
        rel.append(round(float(weights.mean()), 2) if len(weights) else 0.0)
        w = sum(r for _, r in scored)
        sent.append(round(sum(s * r for s, r in scored) / w, 2) if w > 0 else None)
        counts = Counter(c["category"] for x in a for c in x["catalysts"])
        cats.append([k for k, _ in counts.most_common(3)])
        labelled = [c for x in a for c in x["catalysts"] if c["summary"]]
        summaries.append(labelled[0]["summary"] if labelled else title_of.get(ids[0], ""))
    return stories.assign(relevance=rel, sentiment=sent, catalysts=cats, summary=summaries)


# ── Agent 4: Event attribution ──
STOCK_SPECIFIC_Z = 2.0  # |abnormal z| at or above this: the stock's own move
MARKET_WIDE_Z = 1.0     # |abnormal z| below this on a big raw move: the benchmark explains it


def move_type(f) -> str:
    """Classify the session from prices, so the model doesn't have to infer it."""
    if pd.isna(f["abn_z"]):
        return "unknown"
    if abs(f["abn_z"]) >= STOCK_SPECIFIC_Z:
        return "stock-specific"
    if pd.notna(f["ret_z"]) and abs(f["ret_z"]) >= STOCK_SPECIFIC_Z:
        return "market-wide" if abs(f["abn_z"]) < MARKET_WIDE_Z else "mixed"
    return "modest"


_MOVE_TYPE_TEXT = {
    "stock-specific": "stock-specific: the stock moved far more than the benchmark explains, so look for company news",
    "mixed": "mixed: the market moved a lot and the stock moved noticeably more or less than it",
    "market-wide": "market-wide: the benchmark explains most of the move",
    "modest": "modest: neither the stock's own move nor the market's was unusually large",
    "unknown": "unknown",
}


def format_event_data(row, feats: pd.DataFrame) -> str:
    f = feats.loc[row["date"]]
    direction = "UP" if f["abn_ret"] > 0 else "DOWN" if f["abn_ret"] < 0 else "FLAT"
    parts = [f"Events: {', '.join(row['labels'])}",
             f"Direction (abnormal return): {direction}",
             f"Move type (computed from prices): {_MOVE_TYPE_TEXT[move_type(f)]}"]
    if pd.notna(f["ret"]):
        parts.append(f"Stock return {f['ret']:+.2%}; benchmark {f['bench_ret']:+.2%}; "
                     f"abnormal return {f['abn_ret']:+.2%} ({f['abn_z']:+.1f}σ)")
    if pd.notna(f["vol_ratio"]):
        parts.append(f"Volume {f['vol_ratio']:.1f}× its 50-day average; close {f['close']:.2f}")
    return "\n".join(parts)


def format_stories(stories: pd.DataFrame) -> str:
    lines = []
    for s in stories.itertuples():
        sentiment = "n/a" if s.sentiment is None or pd.isna(s.sentiment) else f"{s.sentiment:+.2f}"
        lines.append(f"[{s.story_id.split('-')[-1]}] {s.trading_date.date()} · {s.n_articles} articles · "
                     f"catalysts: {', '.join(s.catalysts) or 'none'} · sentiment {sentiment}\n"
                     f"    {s.headline}\n    {s.summary}")
    return "\n".join(lines)


class EventAttributionAgent:
    """Explains each price-event day using the stories published just before it."""

    max_tokens = 350

    def __init__(self, llm, cfg: Config):
        self.llm = llm
        self.cfg = cfg

    def candidate_stories(self, date: pd.Timestamp, stories: pd.DataFrame, trading_days: pd.DatetimeIndex):
        pos = trading_days.searchsorted(date)
        lo = trading_days[max(pos - self.cfg.attribution_days_before, 0)]
        hi = trading_days[min(pos + self.cfg.attribution_days_after, len(trading_days) - 1)]
        window = stories[(stories["trading_date"] <= hi) & (stories["last_date"] >= lo)]
        window = window[window["relevance"] >= 0.3]
        salience = window["n_articles"] * (0.5 + window["relevance"])
        return window.assign(_s=salience).sort_values("_s", ascending=False).head(self.cfg.max_stories_per_event)

    def attribute(self, ticker: str, event_days: pd.DataFrame, stories: pd.DataFrame, feats: pd.DataFrame,
                  trading_days: pd.DatetimeIndex) -> Tuple[Dict[str, Dict], Dict]:
        days = event_days[event_days["significance"] >= self.cfg.min_event_significance]
        days = days.sort_values("significance", ascending=False)
        if self.cfg.max_events_per_ticker:
            days = days.head(self.cfg.max_events_per_ticker)
        results, prompts, pending = {}, [], []
        for row in days.to_dict("records"):
            key = str(row["date"].date())
            cands = self.candidate_stories(row["date"], stories, trading_days)
            event_data = format_event_data(row, feats)
            base = {
                "date": key, "labels": row["labels"], "significance": row["significance"],
                "move_type": move_type(feats.loc[row["date"]]),
                "abn_ret": None if pd.isna(row["abn_ret"]) else float(row["abn_ret"]),
                "event_data": event_data, "stories": list(cands["story_id"]),
                "story_sentiment": {sid: None if pd.isna(t) else float(t)
                                    for sid, t in zip(cands["story_id"], cands["sentiment"])},
                "stories_text": format_stories(cands),
            }
            if base["move_type"] in ("modest", "unknown"):
                # No unusual daily move (e.g. a new high on a quiet day): there is no single day's news to find.
                results[key] = {**base, "explanation": "No unusual move that day; part of a broader trend.",
                                "primary_driver": "gradual", "cited_stories": [], "cited_tone": None,
                                "confidence": 0.0, "invalid_citations": 0, "driver_conflict": False}
                continue
            if cands.empty:
                results[key] = {**base, "explanation": "No news stories about the company in the window.",
                                "primary_driver": "no news", "cited_stories": [], "cited_tone": None,
                                "confidence": 0.0, "invalid_citations": 0, "driver_conflict": False}
                continue
            prompts.append(EVENT_ATTRIBUTION_PROMPT.format(
                ticker=ticker, date=key, event_data=event_data, days_before=self.cfg.attribution_days_before,
                stories=base["stories_text"], categories=", ".join(CATALYST_CATEGORIES)))
            pending.append(base)

        for base, parsed in zip(pending, generate_json(self.llm, prompts, self.max_tokens,
                                                       schema=EVENT_ATTRIBUTION_SCHEMA)):
            short_to_full = {sid.split("-")[-1]: sid for sid in base["stories"]}
            cited_raw = [str(c).strip("[] ") for c in parsed.get("cited_stories") or []]
            cited = [short_to_full[c] for c in cited_raw if c in short_to_full]
            driver = str(parsed.get("primary_driver") or "unexplained").strip().lower()
            if driver not in ("market-wide", "unexplained"):
                driver = normalize_catalyst(driver)
            # Only a move the prices classify as market-wide can be market-wide; otherwise keep the text but
            # don't count it as explained.
            conflict = driver == "market-wide" and base["move_type"] != "market-wide"
            if conflict:
                driver = "unexplained"
            tones = [t for sid, t in base["story_sentiment"].items() if sid in cited and t is not None]
            results[base["date"]] = {
                **base,
                "cited_tone": round(float(np.mean(tones)), 2) if tones else None,
                "explanation": str(parsed.get("explanation") or "").strip(),
                "primary_driver": driver if parsed else "unexplained",
                "cited_stories": cited,
                "confidence": _clamp(parsed.get("confidence"), 0.0, 1.0) or 0.0,
                "invalid_citations": len(cited_raw) - len(cited),
                "driver_conflict": conflict,
                "parse_failed": not parsed,
            }
        gradual = sum(r["primary_driver"] == "gradual" for r in results.values())
        stats = {"event_days": len(days), "gradual_event_days": gradual,
                 "events_without_news": len(days) - len(pending) - gradual}
        print(f"   🧭 {ticker}: explained {len(pending)} event days with a real move "
              f"({gradual} gradual, {stats['events_without_news']} with no news)")
        return results, stats


# ── Agent 5: Reports ──
def _pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:+.2f}%"


def format_performance(summary: Dict, benchmark: str) -> str:
    if not summary:
        return "No price data for this period."
    excess = summary["excess_return_pct"]
    vs = ("in line with" if pd.isna(excess) or abs(excess) < 0.05 else
          f"{'outperformed' if excess > 0 else 'underperformed'} {benchmark} by {abs(excess):.2f} percentage points")
    return (f"First session {summary['start_date']} close {summary['start_close']} → last session "
            f"{summary['end_date']} close {summary['end_close']} ({_pct(summary['return_pct'])}); "
            f"{benchmark} {_pct(summary['benchmark_return_pct'])}; the stock {vs}.\n"
            f"Period high {summary['period_high']} on {summary['period_high_date']}; "
            f"low {summary['period_low']} on {summary['period_low_date']}.")


def format_events(attributions: List[Dict], limit: int = 10) -> str:
    if not attributions:
        return "No notable price events."
    top = sorted(attributions, key=lambda a: -a["significance"])[:limit]
    return "\n".join(
        f"- {a['date']}: {', '.join(a['labels'])} (gradual move, no single-day driver)." if a["primary_driver"] == "gradual" else
        f"- {a['date']}: {', '.join(a['labels'])} "
        f"(abnormal return {_pct(None if a['abn_ret'] is None else a['abn_ret'] * 100)}). "
        f"Driver: {a['primary_driver']}. {a['explanation']}"
        for a in sorted(top, key=lambda a: a["date"])
    )


def theme_table(analyses: List[Dict]) -> List[Dict]:
    relevant = [a for a in analyses if not a["parse_failed"] and a["relevance"] >= RELEVANT]
    rows = {}
    for a in relevant:
        for cat in {c["category"] for c in a["catalysts"]}:
            r = rows.setdefault(cat, {"category": cat, "articles": 0, "sent": []})
            r["articles"] += 1
            if a["sentiment"] is not None:
                r["sent"].append(a["sentiment"])
    total = len(relevant) or 1
    out = [{"category": r["category"], "articles": r["articles"], "share": r["articles"] / total,
            "avg_sentiment": float(np.mean(r["sent"])) if r["sent"] else None} for r in rows.values()]
    return sorted(out, key=lambda r: -r["articles"])


def format_themes(table: List[Dict], limit: int = 8) -> str:
    if not table:
        return "No relevant articles."
    return "\n".join(
        f"- {r['category']}: {r['share']:.0%} of relevant articles ({r['articles']}), avg sentiment "
        f"{'n/a' if r['avg_sentiment'] is None else format(r['avg_sentiment'], '+.2f')}"
        for r in table[:limit])


def sentiment_series(articles: pd.DataFrame, analyses: Dict[str, Dict], freq: str) -> pd.DataFrame:
    """Relevance-weighted mean sentiment per period (by trading date)."""
    rows = [(r.trading_date, analyses[r.article_id]["sentiment"], analyses[r.article_id]["relevance"])
            for r in articles.itertuples()
            if analyses.get(r.article_id) and analyses[r.article_id]["sentiment"] is not None
            and analyses[r.article_id]["relevance"] >= RELEVANT and pd.notna(r.trading_date)]
    if not rows:
        return pd.DataFrame(columns=["period", "sentiment", "articles"])
    d = pd.DataFrame(rows, columns=["date", "s", "w"])
    d["period"] = d["date"].dt.to_period(freq).dt.start_time
    g = d.assign(sw=d["s"] * d["w"]).groupby("period")
    return pd.DataFrame({"sentiment": g["sw"].sum() / g["w"].sum(), "articles": g.size()}).reset_index()


def format_trend(series: pd.DataFrame, freq_name: str) -> str:
    if series.empty:
        return "No sentiment data."
    return "\n".join(f"- {freq_name} of {p.date()}: {s:+.2f} ({n} articles)"
                     for p, s, n in zip(series["period"], series["sentiment"], series["articles"]))


def format_top_stories(stories: pd.DataFrame, limit: int = 8) -> str:
    if stories.empty:
        return "No stories."
    top = stories[stories["relevance"] >= 0.3].nlargest(limit, "n_articles")
    return "\n".join(f"- {s.trading_date.date()} ({s.n_articles} articles): {s.headline}" for s in top.itertuples())


class ReportAgent:
    """One briefing per time range, written from precomputed figures (the model never does arithmetic)."""

    max_tokens = 900

    def __init__(self, llm, cfg: Config, company_names: Dict[str, str]):
        self.llm = llm
        self.cfg = cfg
        self.company_names = company_names

    def build(self, ticker: str, time_range: str, as_of: datetime, articles: pd.DataFrame,
              analyses: Dict[str, Dict], stories: pd.DataFrame, attributions: Dict[str, Dict],
              market: MarketData) -> Dict:
        start, end = range_bounds(time_range, as_of)
        arts = between(articles, start, end, col="parsed_date") if not articles.empty else articles
        st = between(stories, start, end, col="trading_date") if not stories.empty else stories
        attr = [a for a in attributions.values() if start < pd.Timestamp(a["date"]) <= end]
        window_analyses = [analyses[a] for a in arts["article_id"] if analyses.get(a)] if not arts.empty else []
        days = TIME_RANGE_DELTAS[time_range].days
        freq, freq_name = ("D", "Day") if days <= 7 else ("W", "Week") if days <= 31 else ("M", "Month")
        themes = theme_table(window_analyses)
        in_window = arts[arts["trading_date"] <= pd.Timestamp(end)] if not arts.empty else arts
        trend = sentiment_series(in_window, analyses, freq) if not in_window.empty else pd.DataFrame()
        summary = price_summary(market.prices[ticker], market.benchmark, start, end)
        fields = dict(
            performance=format_performance(summary, self.cfg.benchmark),
            events=format_events(attr),
            themes=format_themes(themes),
            sentiment_trend=format_trend(trend, freq_name),
            top_stories=format_top_stories(st),
        )
        prompt = REPORT_PROMPT.format(
            company=self.company_names.get(ticker, ticker), ticker=ticker, time_range=time_range,
            start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"),
            as_of=as_of.strftime("%Y-%m-%d"), article_count=len(arts), story_count=len(st), **fields)
        source = "\n\n".join(f"{k.upper()}:\n{v}" for k, v in fields.items())
        return {"time_range": time_range, "start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"),
                "articles": len(arts), "stories": len(st), "price_summary": summary, "themes": themes,
                "prompt": prompt, "report_source": source}

    def write(self, specs: List[Dict]) -> List[Dict]:
        reports = self.llm.generate_batch([s["prompt"] for s in specs], max_tokens=self.max_tokens)
        return [{**{k: v for k, v in s.items() if k != "prompt"}, "report": r} for s, r in zip(specs, reports)]
