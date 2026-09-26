# MarketSensAI

**Explains stock moves from the news, grounded in market data.**

For any set of tickers, MarketSensAI reads *every* news article over the past year and detects every notable price event from exchange data. It then explains each move with the stories that drove it and writes investor briefings. Prices say what happened, news says why, and the evaluation checks the news against the prices.

> CS6170 Final Project — Northeastern University
> Anandavardhana Hegde · Ayush Sharma · Yogesh Thakku Suresh

## How it works

```
                    ┌─> analyze_articles  LLM: relevance, sentiment, catalysts, price claims (every article, cached)
    retrieve_news ──┼─> cluster_stories   embeddings: articles covering the same news → one story
                    └─> detect_events     prices: highs/lows, abnormal moves vs. SPY, gaps, volume, trend, drawdowns
                                 │
                                 └─> attribute_events  LLM: why did each event day happen? (cites stories)
                                         └─> write_reports  LLM: one briefing per time range
```

- **News:** Finnhub company news (about a year of history on the free tier) plus yfinance's latest headlines. Stories that don't name the company are dropped, and duplicates are removed. Each article is mapped to the trading session it could first affect: published after the 4pm ET close means the next session.
- **Price events** (`taxonomy.EVENT_TYPES`, from yfinance OHLCV):
  - 4/13/26/52-week, 16-month and all-time highs and lows
  - large gains/drops in *abnormal* return (market-model beta vs. SPY), plus market-wide rallies and selloffs
  - gaps, volume spikes, breakouts/breakdowns, golden/death crosses, corrections and bear markets
  - each day gets a significance score
- **Catalysts** (`taxonomy.CATALYSTS`): earnings, guidance, analyst, product, technology, deal, legal/regulatory, leadership, capital, operations, macro, market, peers. The model can also use "other" with its own label; nothing is forced into a fixed box.
- **Stories:** MiniLM embeddings with similarity-and-date clustering. Forty outlets rewriting one wire story count as one story with forty sources.
- **LLM:** Qwen2.5-7B-Instruct on vLLM (the AWQ 4-bit build on 16 GB GPUs):
  - batched generation with JSON-schema-constrained decoding and greedy sampling
  - prompts ordered so the shared instructions are prefix-cached
  - results cached per article on disk, so runs resume after interruptions and reruns only process new articles

## Running on Colab (recommended)

Open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) on a GPU runtime (A100 or L4 recommended; T4 works but is slower) and run the cells top to bottom. The Finnhub key is read from `.env`, from Colab Secrets (`FINNHUB_API_KEY`), or asked for. Set `QUICK_TEST = True` for a 300-article smoke test first.

It also runs from VS Code with the [Colab extension](https://github.com/googlecolab/colab-vscode): choose *Select Kernel → Colab* and pick a GPU. The kernel runs on Colab's machine, so the code has to come from GitHub or Google Drive.

## CLI

```bash
pip install -e ".[vllm,png]"                       # GPU machine / Colab
echo "FINNHUB_API_KEY=..." > .env                  # free key at finnhub.io

marketsensai fetch --tickers AAPL NVDA             # news + prices only (no GPU)
marketsensai run --tickers AAPL NVDA --png         # full pipeline, every article in range
marketsensai run --max-articles 300                # quick test
marketsensai run --as-of 2026-09-25 --corpus outputs/articles.json   # reproduce a past run
marketsensai ask "Why did NVDA drop in June?" --ticker NVDA
```

Other flags: `--time-ranges 1W 1M 6M 1Y`, `--benchmark SPY`, `--backend hf` (transformers 4-bit, install `.[hf]`), `--model <any HF chat model>`, `--source kaggle --data <csv>` (the paper's static dataset).

Outputs (in `--output-dir`, default `outputs/`):

| File | Contents |
|---|---|
| `articles.json` | the fetched corpus and its as-of date |
| `articles_enriched.csv` | every article with trading day, story, relevance, sentiment, catalysts, price claims |
| `pipeline_results.json` | per ticker: briefings (with their exact source data), every event explanation, stories |
| `eval_results.json` | summary metrics plus per-ticker detail |
| `qa_results.json` | RAG demo answers with sources |
| `review/` | Rating sheets for people: explanations and briefings to score, claim-verification misses to label |
| `visualizations/` | HTML charts (PNGs with `--png`), and a CSV of event explanations per ticker |
| `cache/` | per-article LLM analyses and embeddings (safe to delete; rebuilt on demand) |

## Evaluation

| Metric | How it's measured |
|---|---|
| Sentiment ↔ returns | Spearman ρ (with 95% CI and p-value, per ticker and pooled) between daily news sentiment and same-day / next-day abnormal returns; sign agreement on large-move days |
| Catalyst impact | Mean abnormal return on days each catalyst is in the news, split by news tone, and how often the tone called the direction |
| Event explanations | Share of price-event days with news in the window and explained by it; whether market-wide moves are called market-wide; citation validity |
| Placebo controls | Every explained event day is re-explained with news from a random other week and with the move's direction mirrored; the real explained rate should be well above both |
| Explanation quality | LLM judge: groundedness in the cited stories and plausibility (sample of the most significant events) |
| Claim verification | Price milestones stated in articles ("record high", "52-week low") checked against actual prices |
| Briefing quality | Every number and date in a briefing must appear in its source data; plus an LLM judge of hallucination, faithfulness and relevancy against that data |
| RAG | Judge scores for answers, and whether out-of-corpus tickers are declined |
| Throughput | Articles per minute; time vs. an assumed manual reading rate |

## Project layout

```
src/marketsensai/
├── config.py         # Config dataclass (backend, model, tickers, ranges, thresholds)
├── taxonomy.py       # Price-event and catalyst registries + normalization
├── news.py           # Finnhub + yfinance news, relevance filter, dedupe
├── prices.py         # yfinance prices, abnormal returns, price-event detection
├── data.py           # Corpus build/save/load, Kaggle loader, trading-day alignment
├── embeddings.py     # MiniLM embeddings, cached
├── stories.py        # Story clustering
├── cache.py          # Append-only JSONL cache for LLM results
├── prompts.py        # Prompts and JSON schemas
├── llm.py            # vLLM / transformers backends, constrained JSON generation
├── agents.py         # Retrieval, article analyst, stories, event attribution, reports
├── graph.py          # LangGraph orchestrator
├── rag.py            # Retrieval-augmented Q&A over articles + event explanations
├── evaluation.py     # Metrics above
├── visualization.py  # Plotly figures
└── cli.py            # `marketsensai fetch | run | ask`
notebooks/
├── colab_run.ipynb             # Run everything on a Colab GPU
└── MarketSensAI_original.ipynb # The original notebook (with outputs)
tests/                # Unit + end-to-end tests with a fake LLM, embedder and market (no GPU, no network)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Limitations

- **Summaries, not full articles:** Finnhub and yfinance return headlines and summaries, so analysis sees what a summary says.
- **History limit:** the free Finnhub tier covers about one year.
- **Correlation, not causation:** explanations attribute moves to the news published around them. The groundedness and plausibility scores measure how well-supported that attribution is, not that the news caused the move.
- **Self-judging:** the judge is the same model that wrote the outputs. Treat its scores as a rough signal; a stronger or separate judge is better.
- **Time-to-insight** compares against an assumed manual reading rate, not a measured one.
