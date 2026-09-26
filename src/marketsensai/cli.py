"""Command-line entry point.

    marketsensai fetch --tickers AAPL NVDA                 # news + prices only, no GPU needed
    marketsensai run --tickers AAPL NVDA --time-ranges 1M 1Y
    marketsensai ask "Why did AAPL drop in June?" --ticker AAPL
"""

import argparse
import json
import os
from datetime import datetime

from .config import Config
from .data import TIME_RANGE_DELTAS


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=value lines from a .env file into os.environ; variables already set win."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _config_from_args(args) -> Config:
    cfg = Config(
        news_source=args.source,
        tickers=[t.upper() for t in args.tickers],
        benchmark=args.benchmark.upper(),
        time_ranges=args.time_ranges,
        as_of=datetime.strptime(args.as_of, "%Y-%m-%d") if args.as_of else None,
        data_path=args.data,
        max_articles_per_ticker=args.max_articles,
        output_dir=args.output_dir,
    )
    if getattr(args, "backend", None):  # only run/ask take LLM options
        cfg.llm_backend = args.backend
        cfg.llm_model_name = args.model
    cfg.make_dirs()
    return cfg


def _dump(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _corpus(cfg: Config, corpus_path: str = None, refresh: bool = True):
    """Load a saved corpus if one is given (or refresh is off and one exists), otherwise build and save it."""
    from .data import build_corpus, load_corpus, save_corpus

    path = corpus_path or cfg.corpus_path
    if corpus_path or (not refresh and os.path.exists(path)):
        if not os.path.exists(path):
            raise SystemExit(f"No corpus at {path}. Run `marketsensai fetch` first.")
        df, as_of = load_corpus(path)
        missing = set(cfg.tickers) - set(df["ticker"])
        if missing:
            print(f"   ⚠️ Corpus has no articles for {sorted(missing)}")
        df = df[df["ticker"].isin(cfg.tickers)].reset_index(drop=True)
        print(f"📂 Loaded {len(df):,} articles from {path} (as of {as_of.date()})")
        return df, as_of

    if cfg.news_source == "kaggle" and not os.path.exists(cfg.data_path):
        raise SystemExit(f"Dataset not found at {cfg.data_path}. Download the Kaggle CSV and pass --data.")
    print(f"📥 Building corpus from {cfg.news_source} news...")
    df, as_of = build_corpus(cfg)
    save_corpus(df, as_of, cfg.corpus_path)
    print(f"💾 Saved {len(df):,} articles to {cfg.corpus_path} (as of {as_of.date()})")
    return df, as_of


def _market(cfg: Config):
    from .prices import load_market

    return load_market(cfg.tickers, cfg.benchmark, cfg.event_rules)


def _company_names(tickers):
    from .news import company_name

    return {t: company_name(t) for t in tickers}


def cmd_fetch(args) -> None:
    from .data import range_bounds
    from .prices import between

    cfg = _config_from_args(args)
    df, as_of = _corpus(cfg)
    market = _market(cfg)
    longest = max(cfg.time_ranges, key=TIME_RANGE_DELTAS.get)
    start, end = range_bounds(longest, as_of)
    for t in cfg.tickers:
        n = int((df["ticker"] == t).sum())
        print(f"   {t}: {n:,} articles, {len(between(market.event_days[t], start, end))} price-event days "
              f"in the last {longest}")


def cmd_run(args) -> None:
    from . import evaluation, visualization
    from .embeddings import embed_articles, load_embed_model
    from .graph import build_graph, enriched_articles, run_pipeline, serializable
    from .llm import load_llm
    from .rag import build_rag, demo_questions

    cfg = _config_from_args(args)
    df_articles, as_of = _corpus(cfg, args.corpus)
    market = _market(cfg)
    names = _company_names(cfg.tickers)
    llm = load_llm(cfg)
    embed_model = load_embed_model(cfg.embed_model_name)

    graph = build_graph(df_articles, as_of, market, llm, embed_model, cfg, names)
    results, run_info = run_pipeline(graph, cfg.tickers, as_of)
    enriched = enriched_articles(results)
    enriched.to_csv(os.path.join(cfg.output_dir, "articles_enriched.csv"), index=False)
    _dump(serializable(results, run_info), os.path.join(cfg.output_dir, "pipeline_results.json"))

    qa_results = None
    if not args.skip_rag:
        corpus = df_articles[df_articles["ticker"].isin(cfg.tickers)].reset_index(drop=True)
        emb = embed_articles(corpus, embed_model, cfg.embed_model_name, cfg.cache_dir)
        explanations = {t: list(r["attributions"].values()) for t, r in results.items()}
        rag = build_rag(corpus, emb, explanations, llm, embed_model, as_of, cfg.top_k_retrieval,
                        cfg.recency_weight, cfg.recency_half_life_days)
        print("\n💬 RAG Q&A Demo:")
        qa_results = []
        for q, t in demo_questions(cfg.tickers):
            result = rag.answer(q, ticker=t)
            print(f"\n   Q: {q}\n   A: {result['answer'][:400]}")
            qa_results.append(result)
        _dump(qa_results, os.path.join(cfg.output_dir, "qa_results.json"))

    eval_results = {}
    if not args.skip_eval:
        eval_results = evaluation.run_evaluation(results, run_info, enriched, market, llm, cfg, qa_results)
        _dump(eval_results, os.path.join(cfg.output_dir, "eval_results.json"))

    if not args.skip_viz:
        visualization.make_all(results, eval_results, enriched, market, as_of, cfg.time_ranges, cfg.viz_dir,
                               png=args.png)

    print(f"\n🎉 Done. Outputs written to {cfg.output_dir}")


def cmd_ask(args) -> None:
    from .embeddings import embed_articles, load_embed_model
    from .llm import load_llm
    from .rag import build_rag

    cfg = _config_from_args(args)
    df_articles, as_of = _corpus(cfg, args.corpus, refresh=False)
    explanations = {}
    results_path = os.path.join(cfg.output_dir, "pipeline_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            explanations = {t: r["event_explanations"] for t, r in json.load(f)["tickers"].items()}
    llm = load_llm(cfg)
    embed_model = load_embed_model(cfg.embed_model_name)
    emb = embed_articles(df_articles, embed_model, cfg.embed_model_name, cfg.cache_dir)
    rag = build_rag(df_articles, emb, explanations, llm, embed_model, as_of, cfg.top_k_retrieval,
                    cfg.recency_weight, cfg.recency_half_life_days)

    result = rag.answer(args.question, ticker=args.ticker.upper() if args.ticker else None)
    print(f"\nQ: {result['question']}\nA: {result['answer']}\n\nSources:")
    for s in result["sources"]:
        print(f"  - [{s['date']}] ({s['kind']}) {s['title']} (score={s['score']:.3f})")


def main(argv=None) -> None:
    load_dotenv()
    defaults = Config()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", choices=["live", "kaggle"], default=defaults.news_source,
                        help="live = Finnhub + yfinance news (needs FINNHUB_API_KEY); kaggle = the paper's CSV")
    common.add_argument("--tickers", nargs="+", default=defaults.tickers)
    common.add_argument("--benchmark", default=defaults.benchmark, help="Abnormal returns are measured against this")
    common.add_argument("--time-ranges", nargs="+", default=defaults.time_ranges, choices=list(TIME_RANGE_DELTAS))
    common.add_argument("--as-of", default=None,
                        help="End date YYYY-MM-DD for every range (default: today, or the newest Kaggle article)")
    common.add_argument("--data", default=defaults.data_path, help="Kaggle CSV path (--source kaggle)")
    common.add_argument("--max-articles", type=int, default=None,
                        help="Cap articles per ticker for a quick test run (default: all articles in range)")
    common.add_argument("--output-dir", default=defaults.output_dir)

    llm_args = argparse.ArgumentParser(add_help=False)
    llm_args.add_argument("--backend", choices=["vllm", "hf"], default=defaults.llm_backend)
    llm_args.add_argument("--model", default=defaults.llm_model_name, help="Hugging Face model id")
    llm_args.add_argument("--corpus", default=None, help="Reuse a saved articles.json instead of fetching")

    parser = argparse.ArgumentParser(prog="marketsensai",
                                     description="Explains stock moves from the news, grounded in market data")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", parents=[common], help="Fetch news + prices and save the corpus (no LLM)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_run = sub.add_parser("run", parents=[common, llm_args],
                           help="Run the full pipeline, RAG demo, evaluation, and plots")
    p_run.add_argument("--skip-rag", action="store_true")
    p_run.add_argument("--skip-eval", action="store_true")
    p_run.add_argument("--skip-viz", action="store_true")
    p_run.add_argument("--png", action="store_true", help="Also export PNGs (requires kaleido)")
    p_run.set_defaults(func=cmd_run)

    p_ask = sub.add_parser("ask", parents=[common, llm_args], help="Ask a grounded question over the corpus")
    p_ask.add_argument("question")
    p_ask.add_argument("--ticker", default=None, help="Restrict retrieval to one ticker")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
