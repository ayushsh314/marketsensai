"""End-to-end: graph → RAG → evaluation → charts, with a fake LLM, embedder, and market (no GPU, no network)."""

import json
import os

import pandas as pd

from marketsensai import evaluation, visualization
from marketsensai.config import Config
from marketsensai.embeddings import embed_articles
from marketsensai.graph import build_graph, enriched_articles, run_pipeline, serializable
from marketsensai.rag import OUT_OF_CORPUS_TICKER, build_rag, demo_questions

from conftest import AS_OF, JUMP_DAY, FakeEmbedder, synthetic_market

NAMES = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation"}


def _cfg(tmp_path, **kw):
    cfg = Config(tickers=["AAPL", "MSFT"], time_ranges=["1W", "1M", "1Y"], output_dir=str(tmp_path),
                 story_similarity=0.95, **kw)
    cfg.make_dirs()
    return cfg


def _run(articles, llm, cfg, market=None):
    market = market or synthetic_market(cfg.tickers)
    graph = build_graph(articles, AS_OF, market, llm, FakeEmbedder(), cfg, NAMES)
    results, run_info = run_pipeline(graph, cfg.tickers, AS_OF)
    return market, results, run_info


def test_every_article_is_analyzed_once_and_cached(articles, fake_llm, tmp_path):
    cfg = _cfg(tmp_path)
    _, results, run_info = _run(articles, fake_llm, cfg)
    assert run_info["articles"] == len(articles)  # no cap: all 8 articles in the 1Y range
    analysis_calls = fake_llm.calls_starting("analyze this news item")
    assert len(analysis_calls) == len(articles) + 1  # + one retry for the garbled reply
    assert all(s is not None for c, s in zip(fake_llm.calls, fake_llm.schemas) if c in analysis_calls)

    rerun_llm = type(fake_llm)()
    _, _, rerun_info = _run(articles, rerun_llm, cfg)
    assert rerun_llm.calls_starting("analyze this news item") == []  # served from the on-disk cache
    assert rerun_info["articles_newly_analyzed"] == 0


def test_after_close_news_explains_the_next_session(articles, fake_llm, tmp_path):
    _, results, _ = _run(articles, fake_llm, _cfg(tmp_path))
    r = results["AAPL"]
    earnings = r["articles"][r["articles"]["title"].str.contains("earnings")]
    assert set(earnings["trading_date"]) == {JUMP_DAY}  # published after the 4pm close on the 14th
    assert earnings["story_id"].nunique() <= 2  # near-identical rewrites cluster into one story
    jump = r["attributions"][str(JUMP_DAY.date())]
    assert jump["primary_driver"] == "earnings"
    assert jump["cited_stories"] and all(s in jump["stories"] for s in jump["cited_stories"])
    assert jump["invalid_citations"] == 1  # the made-up "S999" was dropped


def test_only_real_moves_with_news_reach_the_llm(articles, fake_llm, tmp_path):
    _, results, _ = _run(articles, fake_llm, _cfg(tmp_path))
    drivers = [a["primary_driver"] for r in results.values() for a in r["attributions"].values()]
    assert "gradual" in drivers, "quiet-day extremes are marked gradual, not explained"
    assert "no news" in drivers, "most synthetic event days have no articles nearby"
    assert all(a["move_type"] not in ("modest", "unknown")
               for r in results.values() for a in r["attributions"].values() if a["primary_driver"] != "gradual")
    assert len(fake_llm.calls_starting("explain what drove")) == sum(d not in ("no news", "gradual") for d in drivers)


def test_reports_and_analysis_outputs(articles, fake_llm, tmp_path):
    _, results, run_info = _run(articles, fake_llm, _cfg(tmp_path))
    reports = results["AAPL"]["reports"]
    assert [r["time_range"] for r in reports] == ["1W", "1M", "1Y"]
    assert [r["articles"] for r in reports] == [0, 4, 7]  # none in Sep 18-25; four since Aug 26
    assert all(r["report"].startswith("**Summary**") for r in reports)
    assert "EVENTS:" in reports[-1]["report_source"] and "THEMES:" in reports[-1]["report_source"]

    enriched = enriched_articles(results)
    roundup = enriched[enriched["title"] == "Markets roundup"].iloc[0]
    assert roundup["relevance"] == 0.1  # kept, but excluded from sentiment and themes
    lawsuit = enriched[enriched["title"] == "Apple faces lawsuit"].iloc[0]
    assert lawsuit["catalysts"] == ["legal_regulatory"]
    json.dumps(serializable(results, run_info), default=str)  # the output file must serialize


def test_evaluation_rag_and_charts(articles, fake_llm, tmp_path):
    cfg = _cfg(tmp_path)
    market, results, run_info = _run(articles, fake_llm, cfg)
    enriched = enriched_articles(results)
    corpus = articles.reset_index(drop=True)
    emb = embed_articles(corpus, FakeEmbedder(), "fake", cfg.cache_dir)
    explanations = {t: list(r["attributions"].values()) for t, r in results.items()}
    rag = build_rag(corpus, emb, explanations, fake_llm, FakeEmbedder(), AS_OF, top_k=3)
    assert sum(m["kind"] == "event" for m in rag.metadata) == sum(
        bool(a["explanation"]) for items in explanations.values() for a in items)
    msft = rag.retrieve("guidance miss", ticker="MSFT")
    assert msft and all(h["ticker"] == "MSFT" for h in msft)
    qa = [rag.answer(q, ticker=t) for q, t in demo_questions(cfg.tickers)]
    assert "not in the corpus" in qa[-1]["answer"] and qa[-1]["ticker"] == OUT_OF_CORPUS_TICKER

    ev = evaluation.run_evaluation(results, run_info, enriched, market, fake_llm, cfg, qa)
    aapl = ev["per_ticker"]["AAPL"]
    assert aapl["coverage"]["articles"] == 7 and aapl["coverage"]["parse_failures"] == 0
    assert aapl["claims"]["verifiable"] == 1  # the "all-time high" claim
    assert aapl["event_explanations"]["event_days"] > 0
    assert 0 < aapl["event_explanations"]["explained"] < 1
    assert ev["summary"]["report_hallucination"] == 5.0
    assert ev["summary"]["explanation_groundedness"] == 4.0
    assert ev["rag"]["out_of_corpus_declined"] == 1
    assert ev["summary"]["reports_passing_number_check"] == 1.0  # the fake briefing has no numbers
    assert aapl["sentiment_alignment"]["days"] == aapl["sentiment_alignment"]["same_day"]["n"]
    assert aapl["attribution_controls"]["events"] > 0
    timing = aapl["sentiment_alignment"]["by_news_timing"]
    assert set(timing) == {"pre_open", "in_session"} and timing["pre_open"]["days"] > 0
    assert "control_shuffled_news_p_value" in ev["summary"] and "pooled_spearman_same_day_pre_open_news" in ev["summary"]
    review = ev["review"]
    assert len(review["briefings_to_rate"]) == 6  # 2 tickers × 3 ranges
    assert review["explanations_to_rate"] and all(x["groundedness_1to5"] == "" for x in review["explanations_to_rate"])

    visualization.make_all(results, ev, enriched, market, AS_OF, cfg.time_ranges, cfg.viz_dir)
    written = set(os.listdir(cfg.viz_dir))
    for name in ["price_events_AAPL", "sentiment_vs_returns", "sentiment_trend", "catalyst_mix",
                 "event_explanations", "judge_scores", "time_to_insight"]:
        assert f"{name}.html" in written
    assert "event_explanations_AAPL.csv" in written


def test_cli_run_and_ask(articles, fake_llm, tmp_path, monkeypatch):
    from marketsensai import cli, embeddings, llm, news, prices
    from marketsensai.data import save_corpus

    corpus = tmp_path / "articles.json"
    save_corpus(articles, AS_OF, str(corpus))
    monkeypatch.setattr(llm, "load_llm", lambda cfg: fake_llm)
    monkeypatch.setattr(embeddings, "load_embed_model", lambda name: FakeEmbedder())
    monkeypatch.setattr(prices, "load_market", lambda tickers, benchmark, rules: synthetic_market(tickers))
    monkeypatch.setattr(news, "company_name", lambda t: NAMES.get(t, t))

    out = tmp_path / "out"
    cli.main(["run", "--tickers", "AAPL", "MSFT", "--time-ranges", "1M", "1Y",
              "--corpus", str(corpus), "--output-dir", str(out)])
    for name in ["pipeline_results.json", "articles_enriched.csv", "qa_results.json", "eval_results.json"]:
        assert (out / name).exists()
    for name in ["explanations_to_rate", "briefings_to_rate", "claim_misses"]:
        assert (out / "review" / f"{name}.csv").exists()
    ev = json.loads((out / "eval_results.json").read_text())
    assert "review" not in ev and "control_explained_shuffled_news" in ev["summary"]
    assert (out / "cache" / "article_analysis.jsonl").exists()

    cli.main(["ask", "What drove Apple higher?", "--ticker", "aapl", "--tickers", "AAPL", "MSFT",
              "--corpus", str(corpus), "--output-dir", str(out)])


def test_market_wide_label_on_a_stock_specific_move_is_overridden(articles, tmp_path):
    from conftest import FakeLLM

    class SaysMarketWide(FakeLLM):
        def _reply(self, prompt):
            reply = super()._reply(prompt)
            return reply.replace('"Earnings"', '"market-wide"') if prompt.lower().startswith("explain what drove") else reply

    _, results, _ = _run(articles, SaysMarketWide(), _cfg(tmp_path))
    jump = results["AAPL"]["attributions"][str(JUMP_DAY.date())]
    assert jump["move_type"] == "stock-specific"
    assert jump["primary_driver"] == "unexplained" and jump["driver_conflict"] is True


def test_rag_recency_boost_prefers_newer_documents(fake_llm):
    import numpy as np
    from marketsensai.rag import RAGQAModule

    emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")  # identical content, different dates
    meta = [{"kind": "article", "ticker": "AAPL", "date": d, "title": d, "url": ""}
            for d in ("2025-09-25", "2026-09-20")]

    class Const:
        def encode(self, texts, **_):
            return np.array([[1.0, 0.0]] * len(texts))

    rag = RAGQAModule(["old", "new"], meta, emb, Const(), fake_llm, AS_OF, top_k=1, recency_weight=0.1)
    assert rag.retrieve("anything", ticker="AAPL")[0]["text"] == "new"


def test_rag_context_carries_precomputed_ages(fake_llm):
    import numpy as np
    from marketsensai.rag import RAGQAModule

    meta = [{"kind": "event", "ticker": "AAPL", "date": "2026-07-31", "title": "t", "url": ""}]

    class Const:
        def encode(self, texts, **_):
            return np.array([[1.0, 0.0]] * len(texts))

    rag = RAGQAModule(["AAPL fell"], meta, np.array([[1.0, 0.0]], "float32"), Const(), fake_llm, AS_OF)
    assert rag.answer("why?", ticker="AAPL")["context"].startswith("(published 56 days ago)")
    assert rag._age("2025-09-25") == "published about 12 months ago"


def test_placebo_controls_break_explanations_for_a_news_and_direction_aware_model(articles, tmp_path):
    from conftest import FakeLLM

    class Aware(FakeLLM):
        """Explains a move only when it is UP and the stories shown include earnings news."""

        def _reply(self, prompt):
            p = prompt.lower()
            if p.startswith("explain what drove"):
                stories = p.split("news stories published")[1].split("using only these stories")[0]
                if "direction (abnormal return): up" not in p or "earnings" not in stories:
                    return json.dumps({"explanation": "No story fits.", "primary_driver": "unexplained",
                                       "cited_stories": [], "confidence": 0.1})
            return super()._reply(prompt)

    llm = Aware()
    cfg = _cfg(tmp_path)
    market, results, _ = _run(articles, llm, cfg)
    r = results["AAPL"]
    c = evaluation.attribution_controls("AAPL", r["attributions"], r["stories"], market.features["AAPL"],
                                        market.trading_days, llm, cfg)
    assert c["events"] == 1 and c["explained_real"] == 1.0  # the earnings-driven jump
    assert c["shuffled_news_events"] == 1 and c["explained_shuffled_news"] == 0.0  # June news can't explain it
    assert c["explained_flipped_direction"] == 0.0  # good news can't explain a drop
    flipped = [p for p in llm.calls_starting("explain what drove") if "direction (abnormal return): down" in p.lower()]
    assert flipped and "Gap Down" in flipped[-1] and "Large Drop" in flipped[-1] and "-8.00%" in flipped[-1]
    assert c["paired"]["shuffled"] == {"pairs": 1, "real_only": 1, "control_only": 0, "p_value": 1.0}
    event = c["per_event"][0]
    assert event["real_driver"] == "earnings" and event["flipped_driver"] == "unexplained"
    gap = market.trading_days.get_loc(pd.Timestamp(event["shuffled_news_from"])) - market.trading_days.get_loc(JUMP_DAY)
    assert abs(gap) >= cfg.control_min_gap_days


def test_report_source_carries_the_period_facts(articles, fake_llm, tmp_path):
    _, results, _ = _run(articles, fake_llm, _cfg(tmp_path))
    rep = results["AAPL"]["reports"][-1]
    assert rep["report_source"].startswith("PERIOD:\n")
    assert f"{rep['articles']} articles" in rep["report_source"]
    assert "Return over the whole period" in rep["report_source"]
