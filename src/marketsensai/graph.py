"""LangGraph orchestrator — one run per ticker.

                    ┌─> analyze_articles (LLM, every article, cached) ─┐
    retrieve_news ──┼─> cluster_stories  (embeddings)                  ├─> attribute_events (LLM) ─> write_reports (LLM)
                    └─> detect_events    (prices)                      ┘

The three middle branches are independent and run in parallel; attribution waits for all of them.
Every per-article result is computed once over the longest time range, and each range's report
is built from the slice of it that falls inside the range.
"""

import time
from datetime import datetime
from typing import Annotated, Any, Dict, List, Tuple, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from .agents import (
    ArticleAnalystAgent,
    EventAttributionAgent,
    NewsRetrievalAgent,
    ReportAgent,
    StoryAgent,
    profile_stories,
)
from .config import Config
from .data import TIME_RANGE_DELTAS, assign_trading_dates, range_bounds
from .prices import MarketData, between


def _merge(a: Dict, b: Dict) -> Dict:
    return {**a, **b}


class TickerState(TypedDict):
    ticker: str
    articles: Any        # pd.DataFrame, with trading_date
    analyses: Dict       # article_id → analysis
    story_of: Any        # pd.Series: article index → story_id
    stories: Any         # pd.DataFrame
    events: Any          # pd.DataFrame of event days in the longest range
    attributions: Dict   # date → explanation
    reports: List[Dict]
    metadata: Annotated[Dict, _merge]  # the parallel branches all write here


def build_graph(df_articles: pd.DataFrame, as_of: datetime, market: MarketData, llm, embed_model,
                cfg: Config, company_names: Dict[str, str]):
    retrieval = NewsRetrievalAgent(df_articles, as_of, cfg.time_ranges, cfg.max_articles_per_ticker)
    analyst = ArticleAnalystAgent(llm, cfg, company_names)
    story_agent = StoryAgent(embed_model, cfg)
    attributor = EventAttributionAgent(llm, cfg)
    reporter = ReportAgent(llm, cfg, company_names)
    longest = max(cfg.time_ranges, key=TIME_RANGE_DELTAS.get)
    start, end = range_bounds(longest, as_of)

    def retrieve_news(state: TickerState) -> Dict:
        articles = retrieval.retrieve(state["ticker"])
        articles = articles.assign(trading_date=assign_trading_dates(articles, market.trading_days))
        return {"articles": articles}

    def analyze_articles(state: TickerState) -> Dict:
        analyses, stats = analyst.analyze(state["ticker"], state["articles"])
        return {"analyses": analyses, "metadata": stats}

    def cluster(state: TickerState) -> Dict:
        story_of, stories = story_agent.cluster(state["articles"])
        return {"story_of": story_of, "stories": stories, "metadata": {"stories": len(stories)}}

    def detect_events(state: TickerState) -> Dict:
        events = between(market.event_days[state["ticker"]], start, end)
        return {"events": events, "metadata": {"event_days": len(events)}}

    def attribute_events(state: TickerState) -> Dict:
        t = state["ticker"]
        stories = profile_stories(state["stories"], state["articles"], state["analyses"])
        attributions, stats = attributor.attribute(t, state["events"], stories, market.features[t],
                                                   market.trading_days)
        return {"stories": stories, "attributions": attributions, "metadata": stats}

    def write_reports(state: TickerState) -> Dict:
        specs = [reporter.build(state["ticker"], tr, as_of, state["articles"], state["analyses"],
                                state["stories"], state["attributions"], market) for tr in cfg.time_ranges]
        return {"reports": reporter.write(specs)}

    workflow = StateGraph(TickerState)
    workflow.add_node("retrieve_news", retrieve_news)
    workflow.add_node("analyze_articles", analyze_articles)
    workflow.add_node("cluster_stories", cluster)
    workflow.add_node("detect_events", detect_events)
    workflow.add_node("attribute_events", attribute_events)
    workflow.add_node("write_reports", write_reports)

    workflow.set_entry_point("retrieve_news")
    for branch in ("analyze_articles", "cluster_stories", "detect_events"):
        workflow.add_edge("retrieve_news", branch)
    workflow.add_edge(["analyze_articles", "cluster_stories", "detect_events"], "attribute_events")
    workflow.add_edge("attribute_events", "write_reports")
    workflow.add_edge("write_reports", END)
    return workflow.compile()


def run_pipeline(graph, tickers: List[str], as_of: datetime) -> Tuple[Dict[str, Dict], Dict]:
    """Run the graph for each ticker. Returns (per-ticker results, run info)."""
    results = {}
    run_start = time.time()
    for ticker in tickers:
        print(f"\n{'=' * 60}\n🚀 {ticker}\n{'=' * 60}")
        t0 = time.time()
        state = graph.invoke({"ticker": ticker, "articles": pd.DataFrame(), "analyses": {},
                              "story_of": pd.Series(dtype=str), "stories": pd.DataFrame(),
                              "events": pd.DataFrame(), "attributions": {}, "reports": [], "metadata": {}})
        state["metadata"]["seconds"] = round(time.time() - t0, 1)
        articles = state["articles"]
        if not articles.empty:
            articles = articles.assign(story_id=state["story_of"].reindex(articles.index))
        state["articles"] = articles
        results[ticker] = state
        for r in state["reports"]:
            print(f"\n📋 {ticker} ({r['time_range']}):\n{'-' * 40}\n{r['report'][:600]}"
                  f"{'...' if len(r['report']) > 600 else ''}")
    run_info = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "pipeline_seconds": round(time.time() - run_start, 1),
        "articles": int(sum(len(r["articles"]) for r in results.values())),
        "articles_newly_analyzed": int(sum(r["metadata"].get("articles_newly_analyzed", 0)
                                           for r in results.values())),
        "analysis_seconds": round(sum(r["metadata"].get("analysis_seconds", 0) for r in results.values()), 1),
    }
    return results, run_info


def enriched_articles(results: Dict[str, Dict]) -> pd.DataFrame:
    """One row per article with its analysis and story, for saving and for evaluation."""
    rows = []
    for t, r in results.items():
        for a in r["articles"].to_dict("records"):
            an = r["analyses"].get(a["article_id"]) or {}
            rows.append({
                "ticker": t, "article_id": a["article_id"], "published_at": a.get("published_at"),
                "trading_date": a["trading_date"], "story_id": a.get("story_id"), "title": a["title"],
                "publisher": a["publisher"], "url": a["url"], "relevance": an.get("relevance"),
                "sentiment": an.get("sentiment"),
                "catalysts": [c["category"] for c in an.get("catalysts", [])],
                "catalyst_labels": [c["label"] for c in an.get("catalysts", [])],
                "price_claims": an.get("price_claims", []),
                "parse_failed": an.get("parse_failed", True),
            })
    return pd.DataFrame(rows)


def serializable(results: Dict[str, Dict], run_info: Dict) -> Dict:
    """Pipeline output for pipeline_results.json (article-level detail goes to articles_enriched.csv)."""
    out = {"run": run_info, "tickers": {}}
    for t, r in results.items():
        stories = r["stories"]
        out["tickers"][t] = {
            "metadata": r["metadata"],
            "reports": r["reports"],
            "event_explanations": sorted(r["attributions"].values(), key=lambda a: a["date"]),
            "stories": stories.drop(columns=["article_ids"], errors="ignore").to_dict("records")
            if not stories.empty else [],
        }
    return out
