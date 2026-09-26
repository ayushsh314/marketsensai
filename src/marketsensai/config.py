"""Global configuration."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .prices import EventRules

PROMPT_VERSION = "v3"  # bump when analysis prompts change, so cached results are recomputed


@dataclass
class Config:
    # LLM
    llm_backend: str = "vllm"  # "vllm" (Colab GPU, batched, constrained JSON) or "hf" (transformers 4-bit)
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_new_tokens: int = 1024
    temperature: float = 0.0  # greedy decoding so runs are reproducible
    seed: int = 42
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.80  # leaves room on the GPU for the embedding model
    llm_batch_size: int = 1024  # articles per batch between cache checkpoints

    # News
    news_source: str = "live"  # "live" (Finnhub + yfinance) or "kaggle" (the paper's static CSV)
    tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "AMZN"])
    benchmark: str = "SPY"  # abnormal returns are measured against this
    finnhub_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("FINNHUB_API_KEY"))
    finnhub_window_days: int = 7  # Finnhub caps results per request, so the lookback is fetched in windows
    data_path: str = "data/Financial_Sentiment_Categorized.csv"  # only used by the kaggle source
    as_of: Optional[datetime] = None  # end of every time range; None = today (live) or the newest article (kaggle)

    # Articles
    max_articles_per_ticker: Optional[int] = None  # None = every article in the longest range
    max_article_chars: int = 4000

    # Time ranges
    time_ranges: List[str] = field(default_factory=lambda: ["1W", "1M", "6M", "1Y"])

    # Stories (clusters of articles covering the same news)
    story_similarity: float = 0.72  # cosine similarity of MiniLM embeddings
    story_window_days: int = 2

    # Price events and their attribution
    event_rules: EventRules = field(default_factory=EventRules)
    attribution_days_before: int = 2  # news this many trading days before an event can explain it
    attribution_days_after: int = 0
    max_stories_per_event: int = 10
    max_events_per_ticker: Optional[int] = None  # None = explain every qualifying event day, most significant first
    min_event_significance: float = 2.0  # below this (e.g. a lone 4-week high on a quiet day) events are charted, not explained

    # Retrieval
    top_k_retrieval: int = 6
    recency_weight: float = 0.1  # similarity bonus for a document published today ...
    recency_half_life_days: int = 90  # ... halving every this many days

    # Evaluation
    manual_minutes_per_article: float = 0.375  # the paper's ~30 min for 80 articles
    price_tolerance: float = 0.02
    claim_window_days: int = 7
    judge_sample_per_ticker: int = 40  # event explanations sent to the LLM judge per ticker

    # Output
    output_dir: str = "outputs"

    @property
    def corpus_path(self) -> str:
        return os.path.join(self.output_dir, "articles.json")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.output_dir, "cache")

    @property
    def viz_dir(self) -> str:
        return os.path.join(self.output_dir, "visualizations")

    @property
    def cache_namespace(self) -> str:
        return f"{self.llm_model_name}|{PROMPT_VERSION}"

    def make_dirs(self) -> None:
        for d in [self.output_dir, self.cache_dir, self.viz_dir]:
            os.makedirs(d, exist_ok=True)
