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

from collections import Counter
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .agents import RELEVANT
from .config import Config
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


def sentiment_alignment(enriched: pd.DataFrame, feats: pd.DataFrame) -> Dict:
    daily = daily_sentiment(enriched)
    if daily.empty:
        return {"days": 0}
    f = feats[["abn_ret", "abn_z"]].copy()
    f["next_abn_ret"] = f["abn_ret"].shift(-1)
    d = daily.join(f, how="inner")
    big = d[(d["abn_z"].abs() >= LARGE_MOVE_Z) & (d["sentiment"] != 0)]
    agree = (np.sign(big["sentiment"]) == np.sign(big["abn_ret"])).sum()
    return {
        "days": len(d),
        "spearman_same_day": spearman(d["sentiment"], d["abn_ret"]),
        "spearman_next_day": spearman(d["sentiment"], d["next_abn_ret"]),
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


# ── Claim verification ──
def verify_claims(enriched: pd.DataFrame, events: pd.DataFrame, hist: pd.DataFrame, cfg: Config) -> Dict:
    """Price milestones stated in relevant articles, checked against actual price events."""
    claims = [(r.trading_date, c) for r in enriched.itertuples()
              if pd.notna(r.trading_date) and (r.relevance or 0) >= RELEVANT for c in (r.price_claims or [])]
    labelled = [(d, c) for d, c in claims if c.get("label")]
    exact = direction = priced = price_hits = 0
    for date, c in labelled:
        lo_d, hi_d = date - timedelta(days=cfg.claim_window_days), date + timedelta(days=1)
        near = events[(events["date"] >= lo_d) & (events["date"] <= hi_d)] if not events.empty else events
        labels = set(near["label"]) if not near.empty else set()
        exact += c["label"] in labels
        direction += EVENT_TYPES[c["label"]].direction in {EVENT_TYPES[l].direction for l in labels}
        if c.get("price") is not None:
            window = hist[(hist.index >= lo_d) & (hist.index <= hi_d)]
            if not window.empty:
                priced += 1
                lo, hi = window["Low"].min(), window["High"].max()
                price_hits += lo * (1 - cfg.price_tolerance) <= c["price"] <= hi * (1 + cfg.price_tolerance)
    return {
        "claims": len(claims),
        "verifiable": len(labelled),
        "label_precision": _ratio(exact, len(labelled)),
        "direction_precision": _ratio(direction, len(labelled)),
        "with_price": priced,
        "price_accuracy": _ratio(price_hits, priced),
    }


# ── Reports, RAG, throughput ──
def judge_reports(results: Dict[str, Dict], llm) -> Dict:
    items = [(t, r) for t, res in results.items() for r in res["reports"] if r["report"]]
    prompts = [REPORT_JUDGE_PROMPT.format(source_text=r["report_source"], summary_text=r["report"]) for _, r in items]
    return {f"{t}_{r['time_range']}": _scores(j, REPORT_DIMS)
            for (t, r), j in zip(items, generate_json(llm, prompts, 512))}


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
    print("\n📊 Running evaluation...")
    per_ticker = {}
    judged_explanations = {}
    for t, r in results.items():
        e = enriched[enriched["ticker"] == t]
        per_ticker[t] = {
            "coverage": coverage(e, r["stories"]),
            "sentiment_alignment": sentiment_alignment(e, market.features[t]),
            "event_explanations": explanation_stats(r["attributions"]),
            "claims": verify_claims(e, market.events[t], market.prices[t], cfg),
        }
        judged_explanations[t] = judge_explanations(r["attributions"], llm, cfg.judge_sample_per_ticker)
        s, x = per_ticker[t]["sentiment_alignment"], per_ticker[t]["event_explanations"]
        print(f"   {t}: sentiment↔abnormal return ρ={s.get('spearman_same_day')} (same day), "
              f"{s.get('spearman_next_day')} (next day) | events explained {x.get('explained')} "
              f"| claims precision {per_ticker[t]['claims']['label_precision']}")

    impact = catalyst_impact(catalyst_days(enriched), market.features)
    reports = judge_reports(results, llm)
    rag = rag_eval(qa_results, llm) if qa_results else None
    tp = throughput(run_info, cfg)

    all_judged = [j for items in judged_explanations.values() for j in items]
    summary = {
        "articles": tp["articles"],
        "relevant_share": _mean(p["coverage"]["relevant_share"] for p in per_ticker.values()),
        "sentiment_spearman_same_day": _mean(p["sentiment_alignment"].get("spearman_same_day")
                                             for p in per_ticker.values()),
        "sentiment_spearman_next_day": _mean(p["sentiment_alignment"].get("spearman_next_day")
                                             for p in per_ticker.values()),
        "large_move_sign_agreement": _mean(p["sentiment_alignment"].get("large_move_sign_agreement")
                                           for p in per_ticker.values()),
        "events_with_news": _mean(p["event_explanations"].get("with_news_in_window") for p in per_ticker.values()),
        "events_explained": _mean(p["event_explanations"].get("explained") for p in per_ticker.values()),
        "explanation_tone_matches_move": _mean(p["event_explanations"].get("cited_tone_matches_direction")
                                               for p in per_ticker.values()),
        **{f"explanation_{d}": _mean(j[d]["score"] for j in all_judged) for d in ATTRIBUTION_DIMS},
        "claim_label_precision": _mean(p["claims"]["label_precision"] for p in per_ticker.values()),
        **{f"report_{d}": _mean(j[d]["score"] for j in reports.values()) for d in REPORT_DIMS},
        **({f"rag_{d}": v for d, v in rag["means"].items()} if rag else {}),
        "articles_per_minute": tp["articles_per_minute"],
    }
    print("\n── Summary ──")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    return {"summary": summary, "per_ticker": per_ticker, "catalyst_impact": impact,
            "explanation_judge": judged_explanations, "report_judge": reports, "rag": rag, "throughput": tp}
