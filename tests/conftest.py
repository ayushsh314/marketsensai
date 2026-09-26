import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from marketsensai.data import ARTICLE_COLUMNS, extract_milestone_hint
from marketsensai.llm import JSON_RETRY_SUFFIX, BaseLLM
from marketsensai.prices import EventRules, MarketData, compute_features, detect_events, summarize_event_days

AS_OF = datetime(2026, 9, 25)
JUMP_DAY = pd.Timestamp("2026-09-15")  # the synthetic stock gaps up 8% on heavy volume


class FakeLLM(BaseLLM):
    """Canned replies keyed on which prompt template is being used."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.schemas = []

    @staticmethod
    def _quoted(p: str) -> str:
        m = re.search(r'"""\n(.*?)\n"""', p, re.DOTALL)
        return m.group(1) if m else ""

    def _reply(self, prompt: str) -> str:
        p = prompt.lower()
        if p.startswith("analyze this news item"):
            text = self._quoted(p)
            if "garbled" in text and JSON_RETRY_SUFFIX.lower() not in p:
                return "Sure! This article is about a product."  # not JSON: forces the retry path
            sentiment = 0.7 if ("surge" in text or "beat" in text) else -0.6 if ("fall" in text or "miss" in text) else 0.0
            catalysts = []
            if "earnings" in text:
                catalysts.append({"category": "earnings", "label": "Q3 earnings beat", "summary": "Earnings beat."})
            if "lawsuit" in text:
                catalysts.append({"category": "Lawsuit", "label": "DOJ suit", "summary": "A lawsuit was filed."})
            if "launch" in text:
                catalysts.append({"category": "product launch", "label": "New phone", "summary": "A launch."})
            claims = [{"event": "all-time high", "price": "$161.00"}] if "all-time high" in text else []
            if "fell 8%" in text:
                claims.append({"event": "fell 8%", "price": ""})
            return json.dumps({"relevance": 0.1 if "unrelated" in text else 0.9, "sentiment": sentiment,
                               "catalysts": catalysts, "price_claims": claims})
        if p.startswith("explain what drove"):
            ids = re.findall(r"^\[(s\d+)\]", p, re.MULTILINE)
            return json.dumps({"explanation": f"Driven by earnings [{ids[0].upper()}].", "primary_driver": "Earnings",
                               "cited_stories": [ids[0].upper(), "S999"], "confidence": 0.8})
        if p.startswith("write an investor briefing"):
            return "**Summary**: test briefing."
        if "question-answering system" in p:
            return json.dumps({d: {"score": 4, "justification": "ok"}
                               for d in ("context_relevance", "answer_relevance", "groundedness")})
        if "an analyst explained a stock move" in p:
            return json.dumps({d: {"score": 4, "justification": "ok"} for d in ("groundedness", "plausibility")})
        if "rate this ai-generated investor briefing" in p:
            return json.dumps({d: {"score": 5, "justification": "ok"}
                               for d in ("hallucination", "faithfulness", "relevancy")})
        return "Grounded answer."

    def _generate_batch(self, prompts, max_tokens, schema):
        self.calls.extend(prompts)
        self.schemas.extend([schema] * len(prompts))
        return [self._reply(p) for p in prompts]

    def calls_starting(self, prefix: str):
        return [c for c in self.calls if c.lower().startswith(prefix)]


class FakeEmbedder:
    """Bag-of-letters embedding: deterministic, and identical texts land on identical vectors."""

    def encode(self, texts, **_):
        out = np.zeros((len(texts), 26), dtype="float32")
        for i, t in enumerate(texts):
            for ch in re.sub(r"[^a-z]", "", t.lower()):
                out[i, ord(ch) - 97] += 1
        return out + 1e-3


def synthetic_prices(end=AS_OF, days=600, seed=0, jump=True) -> pd.DataFrame:
    """A quiet random walk; optionally an 8% gap-up on heavy volume on JUMP_DAY."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=days)
    rets = rng.normal(0.0003, 0.01, days)
    if jump:
        rets[idx.get_loc(JUMP_DAY)] = 0.08
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.full(days, 1_000_000.0)
    if jump:
        j = idx.get_loc(JUMP_DAY)
        open_[j] = close[j - 1] * 1.07
        volume[j] = 5_000_000.0
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.002,
                         "Low": np.minimum(open_, close) * 0.998, "Close": close, "Volume": volume}, index=idx)


def synthetic_market(tickers) -> MarketData:
    rules = EventRules()
    bench = synthetic_prices(seed=99, jump=False)
    market = MarketData(prices={}, benchmark=bench)
    for i, t in enumerate(tickers):
        hist = synthetic_prices(seed=i, jump=(i == 0))
        feats = compute_features(hist, bench, rules)
        events = detect_events(hist, feats, rules)
        market.prices[t], market.features[t], market.events[t] = hist, feats, events
        market.event_days[t] = summarize_event_days(events)
    return market


def make_articles(rows):
    """rows: (ticker, title, body, published ISO timestamp in UTC)."""
    recs = []
    for i, (t, title, body, ts) in enumerate(rows):
        published = pd.Timestamp(ts)
        day = datetime(published.year, published.month, published.day)
        recs.append({
            "article_id": f"test:{i}", "ticker": t, "title": title, "content_clean": f"{title}. {body}",
            "date": day.strftime("%Y-%m-%d"), "parsed_date": day, "published_at": published.isoformat(),
            "source": "test", "publisher": f"pub{i % 3}", "url": f"https://example.com/{i}",
            "milestone_hint": extract_milestone_hint(title),
        })
    return pd.DataFrame(recs, columns=ARTICLE_COLUMNS)


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def articles():
    return make_articles([
        # Three outlets on the earnings beat the evening before the jump → one story, explains JUMP_DAY.
        ("AAPL", "Apple earnings beat, shares surge after hours", "Record quarter.", "2026-09-14T21:00:00Z"),
        ("AAPL", "Apple earnings beat, shares surge after hours", "Record quarter!", "2026-09-14T21:30:00Z"),
        ("AAPL", "Apple earnings beat expectations", "Shares surge.", "2026-09-14T22:00:00Z"),
        ("AAPL", "Apple hits all-time high", "Stock at an all-time high.", "2026-09-16T15:00:00Z"),
        ("AAPL", "Apple faces lawsuit", "A lawsuit could make shares fall.", "2026-08-20T14:00:00Z"),
        ("AAPL", "Apple supplier note", "Garbled product launch note.", "2026-06-10T14:00:00Z"),
        ("AAPL", "Markets roundup", "Unrelated macro chatter.", "2026-03-02T14:00:00Z"),
        ("MSFT", "Microsoft shares fall on guidance miss", "Guidance miss.", "2026-09-10T14:00:00Z"),
    ])
