"""Evaluation, grounded in market data wherever possible.

- coverage: how much of the news the pipeline could use (relevance, catalysts, parse failures)
- sentiment ↔ returns: rank correlation of daily news sentiment with same-day and next-day abnormal
  returns, and sign agreement on large-move days
- catalyst impact: abnormal returns on days each catalyst was in the news, and whether the news
  tone called the direction
- event explanations: share of price events the news explains, market-wide sanity check, citation
  validity, plus an LLM judge of groundedness and plausibility on a sample
- claim verification: price milestones stated in articles, checked against actual prices
- report and RAG quality: LLM judge against the exact inputs
- throughput: articles processed and time taken
"""

import math
import os
import re
from collections import Counter
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .agents import RELEVANT, EventAttributionAgent, format_event_data
from .config import Config
from .data import MARKET_TZ
from .llm import generate_json
from .prices import MarketData
from .prompts import ATTRIBUTION_JUDGE_PROMPT, RAG_JUDGE_PROMPT, REPORT_JUDGE_PROMPT
from .taxonomy import EVENT_TYPES

REPORT_DIMS = ["hallucination", "faithfulness", "relevancy"]
ATTRIBUTION_DIMS = ["groundedness", "plausibility"]
RAG_DIMS = ["context_relevance", "answer_relevance", "groundedness"]
LARGE_MOVE_Z = 2.0


def _ratio(num, den) -> Optional[float]:
    return round(float(num) / den, 3) if den else None


def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None and not pd.isna(v)]
    return round(float(np.mean(values)), 3) if values else None


def spearman(x: pd.Series, y: pd.Series) -> Optional[float]:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 5 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return None
    return round(float(np.corrcoef(d.iloc[:, 0].rank(), d.iloc[:, 1].rank())[0, 1]), 3)


def rank_correlation(x: pd.Series, y: pd.Series) -> Dict:
    """Spearman ρ with a 95% CI and two-sided p-value (Fisher z, with the Fieller et al. variance 1.06/(n−3))."""
    n = len(pd.concat([x, y], axis=1).dropna())
    rho = spearman(x, y)
    if rho is None:
        return {"rho": None, "n": n}
    z = math.atanh(min(max(rho, -0.9999), 0.9999))
    se = math.sqrt(1.06 / (n - 3))
    return {
        "rho": rho,
        "n": n,
        "ci95": [round(math.tanh(z - 1.96 * se), 3), round(math.tanh(z + 1.96 * se), 3)],
        "p_value": round(math.erfc(abs(z) / se / math.sqrt(2)), 4),
    }


def _relevant(enriched: pd.DataFrame) -> pd.DataFrame:
    return enriched[(~enriched["parse_failed"]) & (enriched["relevance"] >= RELEVANT)
                    & enriched["trading_date"].notna()]


# ── Coverage ──
def coverage(enriched: pd.DataFrame, stories: pd.DataFrame) -> Dict:
    rel = _relevant(enriched)
    with_catalyst = rel["catalysts"].apply(len) > 0
    all_cats = [c for cats in rel["catalysts"] for c in cats]
    return {
        "articles": len(enriched),
        "parse_failures": int(enriched["parse_failed"].sum()),
        "relevant_share": _ratio(len(rel), len(enriched)),
        "relevant_with_catalyst": _ratio(with_catalyst.sum(), len(rel)),
        "catalyst_other_share": _ratio(sum(c == "other" for c in all_cats), len(all_cats)),
        "stories": len(stories),
        "articles_per_story": round(len(enriched) / len(stories), 2) if len(stories) else None,
    }


# ── Sentiment vs. returns ──
def daily_sentiment(enriched: pd.DataFrame) -> pd.DataFrame:
    rel = _relevant(enriched).dropna(subset=["sentiment"])
    if rel.empty:
        return pd.DataFrame(columns=["sentiment", "articles"])
    g = rel.assign(sw=rel["sentiment"] * rel["relevance"]).groupby("trading_date")
    return pd.DataFrame({"sentiment": g["sw"].sum() / g["relevance"].sum(), "articles": g.size()})


def sentiment_days(enriched: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Daily news sentiment joined with that session's and the next session's abnormal return."""
    daily = daily_sentiment(enriched)
    if daily.empty:
        return pd.DataFrame(columns=["sentiment", "articles", "abn_ret", "abn_z", "next_abn_ret"])
    f = feats[["abn_ret", "abn_z"]].copy()
    f["next_abn_ret"] = f["abn_ret"].shift(-1)
    return daily.join(f, how="inner")


MARKET_OPEN_MINUTES = 9 * 60 + 30  # 9:30 ET
NEWS_TIMINGS = ["pre_open", "in_session"]


def news_timing(enriched: pd.DataFrame) -> pd.Series:
    """When each article appeared relative to the session it is assigned to.

    "pre_open": before that session opened (overnight, weekend, or after the previous close), so it can't be
    a reaction to that session's move. "in_session": during the session, where coverage often reports the move
    itself. None when only a date is known (e.g. the Kaggle source).
    """
    ts = pd.to_datetime(enriched["published_at"], utc=True, errors="coerce").dt.tz_convert(MARKET_TZ)
    day = ts.dt.tz_localize(None).dt.normalize()
    minutes = ts.dt.hour * 60 + ts.dt.minute
    in_session = (day == pd.to_datetime(enriched["trading_date"])) & (minutes >= MARKET_OPEN_MINUTES)
    labels = np.where(in_session, "in_session", "pre_open").astype(object)
    labels[ts.isna().to_numpy()] = None
    return pd.Series(labels, index=enriched.index)


def correlations(d: pd.DataFrame) -> Dict:
    return {"days": len(d), "same_day": rank_correlation(d["sentiment"], d["abn_ret"]),
            "next_day": rank_correlation(d["sentiment"], d["next_abn_ret"])}


def sentiment_alignment(d: pd.DataFrame) -> Dict:
    if d.empty:
        return {"days": 0}
    big = d[(d["abn_z"].abs() >= LARGE_MOVE_Z) & (d["sentiment"] != 0)]
    agree = (np.sign(big["sentiment"]) == np.sign(big["abn_ret"])).sum()
    same, nxt = rank_correlation(d["sentiment"], d["abn_ret"]), rank_correlation(d["sentiment"], d["next_abn_ret"])
    return {
        "days": len(d),
        "spearman_same_day": same["rho"],
        "spearman_next_day": nxt["rho"],
        "same_day": same,
        "next_day": nxt,
        "large_move_days": len(big),
        "large_move_sign_agreement": _ratio(agree, len(big)),
    }


# ── Catalyst impact ──
HEAVY_COVERAGE = 3  # relevant articles on one catalyst in one session


def catalyst_days(enriched: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, trading day, catalyst): the day's mean sentiment and article count for it."""
    rel = _relevant(enriched)
    rows = rel.explode("catalysts").dropna(subset=["catalysts"])
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "trading_date", "category", "sentiment", "articles"])
    g = rows.groupby(["ticker", "trading_date", "catalysts"])["sentiment"]
    return (pd.DataFrame({"sentiment": g.mean(), "articles": g.size()}).reset_index()
            .rename(columns={"catalysts": "category"}))


def catalyst_impact(cat_days: pd.DataFrame, features: Dict[str, pd.DataFrame], min_days: int = 5,
                    min_articles: int = HEAVY_COVERAGE) -> List[Dict]:
    """Abnormal returns on days a catalyst was heavily covered (most categories appear in some article almost
    every day, so a single mention says little)."""
    if cat_days.empty:
        return []
    cat_days = cat_days[cat_days["articles"] >= min_articles]
    abn = pd.concat({t: f["abn_ret"] for t, f in features.items()}, names=["ticker", "trading_date"])
    d = cat_days.join(abn.rename("abn_ret"), on=["ticker", "trading_date"]).dropna(subset=["abn_ret"])
    baseline = float(abn.abs().mean())
    out = []
    for cat, g in d.groupby("category"):
        if len(g) < min_days:
            continue
        toned = g[g["sentiment"].abs() > 0.1]
        bull, bear = toned[toned["sentiment"] > 0], toned[toned["sentiment"] < 0]
        out.append({
            "category": cat,
            "days": len(g),
            "mean_abs_abn_ret_pct": round(float(g["abn_ret"].abs().mean()) * 100, 3),
            "vs_typical_day": round(float(g["abn_ret"].abs().mean()) / baseline, 2) if baseline else None,
            "bullish_days_mean_abn_ret_pct": round(float(bull["abn_ret"].mean()) * 100, 3) if len(bull) else None,
            "bearish_days_mean_abn_ret_pct": round(float(bear["abn_ret"].mean()) * 100, 3) if len(bear) else None,
            "direction_agreement": _ratio((np.sign(toned["sentiment"]) == np.sign(toned["abn_ret"])).sum(),
                                          len(toned)),
        })
    return sorted(out, key=lambda r: -r["days"])


# ── Event explanations ──
def explanation_stats(attributions: Dict[str, Dict]) -> Dict:
    """Stats over event days with a real move; "gradual" days (no unusual move) have nothing to explain."""
    gradual = sum(a["primary_driver"] == "gradual" for a in attributions.values())
    items = [a for a in attributions.values() if a["primary_driver"] != "gradual"]
    if not items:
        return {"event_days": 0, "gradual_event_days": gradual}
    drivers = Counter(a["primary_driver"] for a in items)
    with_news = [a for a in items if a["primary_driver"] != "no news"]
    explained = [a for a in with_news if a["primary_driver"] not in ("unexplained",)]
    market_labelled = [a for a in items if any(l.startswith("Market-Wide") for l in a["labels"])]
    by_category = {}
    for cat in ["extreme", "move", "gap", "volume", "trend", "drawdown"]:
        rows = [a for a in items if any(EVENT_TYPES[l].category == cat for l in a["labels"])]
        if rows:
            by_category[cat] = {
                "event_days": len(rows),
                "news_explained": _ratio(sum(a["primary_driver"] not in ("no news", "unexplained") for a in rows),
                                         len(rows)),
            }
    toned = [a for a in explained if a.get("cited_tone") is not None and abs(a["cited_tone"]) > 0.05
             and a.get("abn_ret")]
    consistent = sum(np.sign(a["cited_tone"]) == np.sign(a["abn_ret"]) for a in toned)
    cited = sum(len(a["cited_stories"]) for a in with_news)
    invalid = sum(a.get("invalid_citations", 0) for a in with_news)
    return {
        "event_days": len(items),
        "gradual_event_days": gradual,
        "with_news_in_window": _ratio(len(with_news), len(items)),
        "explained": _ratio(len(explained), len(items)),
        "explained_when_news": _ratio(len(explained), len(with_news)),
        "drivers": dict(drivers.most_common()),
        "market_wide_events_called_market_wide": _ratio(
            sum(a["primary_driver"] == "market-wide" for a in market_labelled), len(market_labelled)),
        "invalid_citation_rate": _ratio(invalid, cited + invalid),
        "market_wide_conflicts": sum(a.get("driver_conflict", False) for a in items),
        "cited_tone_matches_direction": _ratio(consistent, len(toned)),
        "by_event_category": by_category,
    }


def _scores(parsed: Dict, dims: List[str]) -> Dict:
    out = {}
    for d in dims:
        entry = parsed.get(d) if isinstance(parsed.get(d), dict) else {}
        try:
            score = float(entry.get("score"))
        except (TypeError, ValueError):
            score = None
        out[d] = {"score": score, "justification": entry.get("justification", "")}
    return out


def judge_explanations(attributions: Dict[str, Dict], llm, sample: int) -> List[Dict]:
    explained = [a for a in attributions.values()
                 if a["primary_driver"] not in ("no news", "gradual") and a["explanation"]]
    top = sorted(explained, key=lambda a: -a["significance"])[:sample]
    prompts = [ATTRIBUTION_JUDGE_PROMPT.format(event_data=a["event_data"], stories=a["stories_text"],
                                               explanation=a["explanation"]) for a in top]
    judged = generate_json(llm, prompts, 300)
    return [{"date": a["date"], **_scores(j, ATTRIBUTION_DIMS)} for a, j in zip(top, judged)]


# ── Attribution controls (placebo tests) ──
_OPPOSITES = [("High", "Low"), ("Large Gain", "Large Drop"), ("Market-Wide Rally", "Market-Wide Selloff"),
              ("Gap Up", "Gap Down"), ("Breakout", "Breakdown"), ("Golden Cross", "Death Cross")]
RETURN_COLS = ["ret", "bench_ret", "abn_ret", "ret_z", "abn_z"]


def flip_label(label: str) -> Optional[str]:
    """The same event in the opposite direction ("52-Week High" → "52-Week Low"); None if it has none."""
    if EVENT_TYPES[label].direction == "neutral":
        return label
    for a, b in _OPPOSITES:
        for x, y in ((a, b), (b, a)):
            if label == x or label.endswith(f" {x}"):
                return label[: len(label) - len(x)] + y
    return None  # Correction, Bear Market: drawdowns have no upward twin


def attribution_controls(ticker: str, attributions: Dict[str, Dict], stories: pd.DataFrame, feats: pd.DataFrame,
                         trading_days: pd.DatetimeIndex, llm, cfg: Config) -> Dict:
    """Re-explain every LLM-explained event day twice, with the evidence deliberately broken.

    - shuffled news: the real market data, but the stories from a random session at least
      cfg.control_min_gap_days trading days away. A model that uses the news should mostly say "unexplained".
    - flipped direction: the real stories, but the move (returns, z-scores, labels) mirrored. Good news
      shouldn't explain a drop, so a direction-aware model should again mostly say "unexplained".
    If either explained rate is close to the real one, "explained" measures willingness to tell a story.
    """
    real = [a for a in attributions.values()
            if a["primary_driver"] not in ("gradual", "no news") and not a.get("parse_failed")]
    if not real or stories.empty:
        return {"events": 0}
    agent = EventAttributionAgent(llm, cfg)
    rng = np.random.default_rng(cfg.seed)
    by_id = stories.set_index("story_id", drop=False)
    pool = trading_days[(trading_days >= stories["trading_date"].min()) & (trading_days <= stories["last_date"].max())]
    pool_pos = trading_days.searchsorted(pool)
    shuffled, shuffled_from, flipped = {}, {}, []  # shuffled: index into real → base (some events find none)
    for i, a in enumerate(real):
        date = pd.Timestamp(a["date"])
        far = pool[np.abs(pool_pos - trading_days.searchsorted(date)) >= cfg.control_min_gap_days]
        for other in rng.permutation(far.to_numpy()):
            cands = agent.candidate_stories(pd.Timestamp(other), stories, trading_days)
            if not cands.empty:
                shuffled[i] = agent.base(a["date"], a, cands, a["event_data"], a["move_type"])
                shuffled_from[i] = str(pd.Timestamp(other).date())
                break
        f = feats.loc[[date]].copy()
        f[RETURN_COLS] = -f[RETURN_COLS]
        labels = [x for x in (flip_label(l) for l in a["labels"]) if x]
        row = {**a, "date": date, "labels": labels, "abn_ret": None if a["abn_ret"] is None else -a["abn_ret"]}
        flipped.append(agent.base(a["date"], row, by_id.loc[a["stories"]], format_event_data(row, f), a["move_type"]))

    def driver(x: Dict) -> str:
        return "unexplained" if x["parse_failed"] else x["primary_driver"]

    shuffled_drivers = dict(zip(shuffled, map(driver, agent.explain(ticker, list(shuffled.values())))))
    flipped_drivers = list(map(driver, agent.explain(ticker, flipped)))
    per_event = [{"date": a["date"], "real_driver": driver(a), "shuffled_news_from": shuffled_from.get(i),
                  "shuffled_driver": shuffled_drivers.get(i), "flipped_driver": flipped_drivers[i]}
                 for i, a in enumerate(real)]
    return {
        "events": len(real),
        "explained_real": _ratio(_explained_count(per_event, "real_driver"), len(real)),
        "shuffled_news_events": len(shuffled),
        "explained_shuffled_news": _ratio(_explained_count(per_event, "shuffled_driver"), len(shuffled)),
        "explained_flipped_direction": _ratio(_explained_count(per_event, "flipped_driver"), len(real)),
        "counts": {"real": [_explained_count(per_event, "real_driver"), len(real)],
                   "shuffled": [_explained_count(per_event, "shuffled_driver"), len(shuffled)],
                   "flipped": [_explained_count(per_event, "flipped_driver"), len(real)]},
        "paired": {kind: _paired(per_event, f"{kind}_driver") for kind in ("shuffled", "flipped")},
        "per_event": per_event,
    }


def _explained_count(per_event: List[Dict], key: str) -> int:
    return sum(e[key] is not None and e[key] != "unexplained" for e in per_event)


def _paired(per_event: List[Dict], key: str) -> Dict:
    """Events explained with the real evidence but not the control, and vice versa, with McNemar's exact test."""
    pairs = [(e["real_driver"] != "unexplained", e[key] != "unexplained") for e in per_event if e[key] is not None]
    real_only = sum(r and not c for r, c in pairs)
    control_only = sum(c and not r for r, c in pairs)
    return {"pairs": len(pairs), "real_only": real_only, "control_only": control_only,
            "p_value": mcnemar_exact(real_only, control_only)}


def mcnemar_exact(b: int, c: int) -> Optional[float]:
    """Two-sided exact McNemar test on the discordant pairs (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return None
    return round(min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n), 4)


# ── Claim verification ──
def verify_claims(enriched: pd.DataFrame, events: pd.DataFrame, hist: pd.DataFrame, cfg: Config,
                  misses: Optional[List[Dict]] = None) -> Dict:
    """Price milestones stated in relevant articles, checked against actual price events.

    Claims whose label matches no event in the window are appended to `misses` (if given) for review.
    """
    claims = [(r, c) for r in enriched.itertuples()
              if pd.notna(r.trading_date) and (r.relevance or 0) >= RELEVANT for c in (r.price_claims or [])]
    labelled = [(r, c) for r, c in claims if c.get("label")]
    exact = direction = priced = price_hits = 0
    for r, c in labelled:
        date = r.trading_date
        lo_d, hi_d = date - timedelta(days=cfg.claim_window_days), date + timedelta(days=1)
        near = events[(events["date"] >= lo_d) & (events["date"] <= hi_d)] if not events.empty else events
        labels = set(near["label"]) if not near.empty else set()
        hit = c["label"] in labels
        exact += hit
        direction += EVENT_TYPES[c["label"]].direction in {EVENT_TYPES[l].direction for l in labels}
        price_ok = None
        if c.get("price") is not None:
            window = hist[(hist.index >= lo_d) & (hist.index <= hi_d)]
            if not window.empty:
                priced += 1
                lo, hi = window["Low"].min(), window["High"].max()
                price_ok = bool(lo * (1 - cfg.price_tolerance) <= c["price"] <= hi * (1 + cfg.price_tolerance))
                price_hits += price_ok
        if not hit and misses is not None:
            misses.append({
                "ticker": r.ticker, "trading_date": str(pd.Timestamp(date).date()), "title": r.title,
                "url": r.url, "claim_as_written": c["event"], "claim_label": c["label"],
                "claim_price": c.get("price"), "price_in_range": price_ok,
                "events_in_window": ", ".join(sorted(labels)),
                "verdict": "",  # for the reviewer: extraction error / loose wording / matching too strict / other
            })
    return {
        "claims": len(claims),
        "verifiable": len(labelled),
        "label_precision": _ratio(exact, len(labelled)),
        "direction_precision": _ratio(direction, len(labelled)),
        "with_price": priced,
        "price_accuracy": _ratio(price_hits, priced),
    }


# ── Reports ──
def judge_reports(results: Dict[str, Dict], llm) -> Dict:
    items = [(t, r) for t, res in results.items() for r in res["reports"] if r["report"]]
    prompts = [REPORT_JUDGE_PROMPT.format(source_text=r["report_source"], summary_text=r["report"]) for _, r in items]
    return {f"{t}_{r['time_range']}": _scores(j, REPORT_DIMS)
            for (t, r), j in zip(items, generate_json(llm, prompts, 512))}


_DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")  # full dates, and months like "2025-12" in monthly trends
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
# Numbers that appear in briefings as names rather than data: "52-week", "50-day", "S&P 500", small counts.
_STRUCTURAL = {13, 16, 26, 50, 52, 200, 500}


def _numbers(text: str) -> List[str]:
    return _NUMBER.findall(_DATE.sub(" ", text))


def number_check(report: str, source: str) -> Dict:
    """Every date and number in a briefing must appear in its source data.

    A number counts as supported if some source number rounds to it ("38.3%" from 38.29, "$341" from
    341.07). Signs are ignored ("down 3.40%" for -3.40%). This catches invented or garbled figures, not
    real figures put in the wrong context.
    """
    src_dates = set(_DATE.findall(source))
    src = {abs(float(n.replace(",", ""))) for n in _numbers(source)} | {float(d[:4]) for d in src_dates}
    bad_dates = sorted({d for d in _DATE.findall(report) if not any(s.startswith(d) for s in src_dates)})
    checked, bad = 0, []
    for token in _numbers(report):
        value = float(token.replace(",", ""))
        if "." not in token and (value < 10 or value in _STRUCTURAL):
            continue
        checked += 1
        places = len(token.split(".")[1]) if "." in token else 0
        if not any(abs(round(s, places) - value) < 1e-9 for s in src):
            bad.append(token)
    return {"numbers": checked, "dates": len(_DATE.findall(report)), "unsupported_numbers": bad,
            "unsupported_dates": bad_dates, "passed": not bad and not bad_dates}


def check_reports(results: Dict[str, Dict]) -> Dict:
    return {f"{t}_{r['time_range']}": number_check(r["report"], r["report_source"])
            for t, res in results.items() for r in res["reports"] if r["report"]}


# ── Human review files ──
def review_items(results: Dict[str, Dict], judged: Dict[str, List[Dict]], claim_misses: List[Dict],
                 cfg: Config) -> Dict[str, List[Dict]]:
    """Blank rating sheets for the team: a random sample of judged explanations (so ratings can be compared
    with the LLM judge by ticker and date), every briefing, and every claim miss."""
    rng = np.random.default_rng(cfg.seed)
    explanations, briefings = [], []
    for t, r in results.items():
        dates = [j["date"] for j in judged.get(t, [])]
        for d in sorted(rng.permutation(dates)[:cfg.review_per_ticker]) if dates else []:
            a = r["attributions"][d]
            explanations.append({
                "review_id": f"E{len(explanations) + 1}", "ticker": t, "date": d, "labels": ", ".join(a["labels"]),
                "market_data": a["event_data"], "stories_shown": a["stories_text"], "explanation": a["explanation"],
                "primary_driver": a["primary_driver"], "groundedness_1to5": "", "plausibility_1to5": "", "notes": "",
            })
        for rep in r["reports"]:
            briefings.append({
                "review_id": f"B{len(briefings) + 1}", "ticker": t, "time_range": rep["time_range"],
                "source_data": rep["report_source"], "briefing": rep["report"], "hallucination_1to5": "",
                "faithfulness_1to5": "", "relevancy_1to5": "", "notes": "",
            })
    return {"explanations_to_rate": explanations, "briefings_to_rate": briefings, "claim_misses": claim_misses}


def save_review(review: Dict[str, List[Dict]], review_dir: str) -> None:
    os.makedirs(review_dir, exist_ok=True)
    for name, rows in review.items():
        pd.DataFrame(rows).to_csv(os.path.join(review_dir, f"{name}.csv"), index=False)
    print(f"📝 Review sheets written to {review_dir}")


# ── RAG, throughput ──
def rag_eval(qa_results: List[Dict], llm) -> Dict:
    answered = [q for q in qa_results if q["sources"]]
    prompts = [RAG_JUDGE_PROMPT.format(question=q["question"], context=q["context"], answer=q["answer"])
               for q in answered]
    judged = [_scores(p, RAG_DIMS) for p in generate_json(llm, prompts, 512)]
    return {
        "per_question": [{"question": q["question"], **s} for q, s in zip(answered, judged)],
        "means": {d: _mean(s[d]["score"] for s in judged) for d in RAG_DIMS},
        "out_of_corpus_declined": sum(not q["sources"] for q in qa_results),
    }


def throughput(run_info: Dict, cfg: Config) -> Dict:
    """Speed of the LLM analysis. Undefined when every article came from the cache (nothing was timed)."""
    seconds = run_info["pipeline_seconds"]
    new, analysis_s = run_info["articles_newly_analyzed"], run_info.get("analysis_seconds", 0)
    fresh = new > 0 and analysis_s > 0
    manual = run_info["articles"] * cfg.manual_minutes_per_article * 60
    return {
        "pipeline_seconds": seconds,
        "articles": run_info["articles"],
        "articles_newly_analyzed": new,
        "analysis_seconds": analysis_s,
        "fully_cached": new == 0,
        "articles_per_minute": round(new / analysis_s * 60, 1) if fresh else None,
        "manual_estimate_seconds": round(manual, 1),
        "manual_minutes_per_article_assumed": cfg.manual_minutes_per_article,
        # Only meaningful when this run analyzed everything itself.
        "speedup_factor": round(manual / seconds, 1) if new == run_info["articles"] and seconds else None,
    }


def run_evaluation(results: Dict[str, Dict], run_info: Dict, enriched: pd.DataFrame, market: MarketData,
                   llm, cfg: Config, qa_results: Optional[List[Dict]] = None) -> Dict:
    """All metrics. The returned "review" entry holds the human rating sheets (see save_review)."""
    print("\n📊 Running evaluation...")
    per_ticker, judged_explanations, sentiment_frames, claim_misses = {}, {}, [], []
    timed_frames = {k: [] for k in NEWS_TIMINGS}
    for t, r in results.items():
        e = enriched[enriched["ticker"] == t]
        days = sentiment_days(e, market.features[t])
        sentiment_frames.append(days)
        timing = news_timing(e)
        timed = {k: sentiment_days(e[timing == k], market.features[t]) for k in NEWS_TIMINGS}
        for k, d in timed.items():
            timed_frames[k].append(d)
        per_ticker[t] = {
            "coverage": coverage(e, r["stories"]),
            "sentiment_alignment": {**sentiment_alignment(days),
                                    "by_news_timing": {k: correlations(d) for k, d in timed.items()}},
            "event_explanations": explanation_stats(r["attributions"]),
            "attribution_controls": attribution_controls(t, r["attributions"], r["stories"], market.features[t],
                                                         market.trading_days, llm, cfg),
            "claims": verify_claims(e, market.events[t], market.prices[t], cfg, claim_misses),
        }
        judged_explanations[t] = judge_explanations(r["attributions"], llm, cfg.judge_sample_per_ticker)
        s, x, c = (per_ticker[t][k] for k in ("sentiment_alignment", "event_explanations", "attribution_controls"))
        print(f"   {t}: sentiment↔abnormal return ρ={s.get('spearman_same_day')} (same day), "
              f"{s.get('spearman_next_day')} (next day) | events explained {x.get('explained')} "
              f"| placebo: shuffled news {c.get('explained_shuffled_news')}, "
              f"flipped direction {c.get('explained_flipped_direction')} "
              f"| claims precision {per_ticker[t]['claims']['label_precision']}")

    def pool(frames: List[pd.DataFrame]) -> Dict:
        frames = [d for d in frames if not d.empty]
        d = pd.concat(frames) if frames else pd.DataFrame(columns=["sentiment", "abn_ret", "next_abn_ret"])
        return correlations(d)

    pooled, pooled_timed = pool(sentiment_frames), {k: pool(f) for k, f in timed_frames.items()}

    def paired_p(kind: str) -> Optional[float]:
        paired = [p["attribution_controls"]["paired"][kind] for p in per_ticker.values()
                  if "paired" in p["attribution_controls"]]
        return mcnemar_exact(sum(x["real_only"] for x in paired), sum(x["control_only"] for x in paired))

    def control_rate(kind: str) -> Optional[float]:
        counts = [p["attribution_controls"]["counts"][kind] for p in per_ticker.values()
                  if "counts" in p["attribution_controls"]]
        return _ratio(sum(k for k, _ in counts), sum(n for _, n in counts))

    impact = catalyst_impact(catalyst_days(enriched), market.features)
    reports = judge_reports(results, llm)
    number_checks = check_reports(results)
    rag = rag_eval(qa_results, llm) if qa_results else None
    tp = throughput(run_info, cfg)

    all_judged = [j for items in judged_explanations.values() for j in items]
    checked = sum(c["numbers"] for c in number_checks.values())
    summary = {
        "articles": tp["articles"],
        "relevant_share": _mean(p["coverage"]["relevant_share"] for p in per_ticker.values()),
        "sentiment_spearman_same_day": _mean(p["sentiment_alignment"].get("spearman_same_day")
                                             for p in per_ticker.values()),
        "sentiment_spearman_next_day": _mean(p["sentiment_alignment"].get("spearman_next_day")
                                             for p in per_ticker.values()),
        "pooled_spearman_same_day": pooled["same_day"],
        "pooled_spearman_next_day": pooled["next_day"],
        # Pre-open news can't be a reaction to that session's move; in-session coverage often is.
        **{f"pooled_spearman_same_day_{k}_news": v["same_day"] for k, v in pooled_timed.items()},
        "large_move_sign_agreement": _mean(p["sentiment_alignment"].get("large_move_sign_agreement")
                                           for p in per_ticker.values()),
        "events_with_news": _mean(p["event_explanations"].get("with_news_in_window") for p in per_ticker.values()),
        "events_explained": _mean(p["event_explanations"].get("explained") for p in per_ticker.values()),
        "control_explained_real": control_rate("real"),
        "control_explained_shuffled_news": control_rate("shuffled"),
        "control_explained_flipped_direction": control_rate("flipped"),
        "control_shuffled_news_p_value": paired_p("shuffled"),  # McNemar exact, real vs. control, pooled
        "control_flipped_direction_p_value": paired_p("flipped"),
        "explanation_tone_matches_move": _mean(p["event_explanations"].get("cited_tone_matches_direction")
                                               for p in per_ticker.values()),
        **{f"explanation_{d}": _mean(j[d]["score"] for j in all_judged) for d in ATTRIBUTION_DIMS},
        "claim_label_precision": _mean(p["claims"]["label_precision"] for p in per_ticker.values()),
        **{f"report_{d}": _mean(j[d]["score"] for j in reports.values()) for d in REPORT_DIMS},
        "report_numbers_supported": _ratio(checked - sum(len(c["unsupported_numbers"])
                                                         for c in number_checks.values()), checked),
        "reports_passing_number_check": _ratio(sum(c["passed"] for c in number_checks.values()), len(number_checks)),
        **({f"rag_{d}": v for d, v in rag["means"].items()} if rag else {}),
        "articles_per_minute": tp["articles_per_minute"],
    }
    print("\n── Summary ──")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    return {"summary": summary, "per_ticker": per_ticker, "catalyst_impact": impact,
            "explanation_judge": judged_explanations, "report_judge": reports, "report_number_check": number_checks,
            "rag": rag, "throughput": tp,
            "review": review_items(results, judged_explanations, claim_misses, cfg)}
