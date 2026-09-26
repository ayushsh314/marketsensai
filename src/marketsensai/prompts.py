"""Prompt templates and JSON schemas for the agents, RAG Q&A, and LLM judges."""

# ── Article analysis (one call per article) ──
# The fixed instructions come first and the article last, so vLLM's prefix cache computes the shared
# part once for all articles instead of once per article.
ARTICLE_ANALYSIS_PROMPT = """Analyze this news item about a company's stock and return JSON with:
- "relevance": 0.0-1.0, how much the item is about the company itself (0 = only mentioned in passing, 1 = entirely about it).
- "sentiment": -1.0 (very bearish) to 1.0 (very bullish) for the company's stock, based only on this item. 0 if neutral or not about the company.
- "catalysts": up to 3 things in the item that could move the stock. For each give "category" (one of the keys below, or "other"), a short specific "label" in your own words (e.g. "iPhone 17 demand beat", "DOJ antitrust suit"), and a one-sentence "summary".
- "price_claims": price milestones the item says the company's own stock actually reached at or just before publication (e.g. "hit an all-time high", "fell to a 52-week low", "broke out"). Give the "event" as written and the "price" if one is stated (else ""). Do NOT include distances from a level ("11% below its 52-week high"), levels it is near, approaching or targeting, past years, or other companies' stocks. Empty list if none.

Catalyst categories:
{catalyst_menu}

Company: {company} ({ticker})
Published: {published}
News item:
\"\"\"
{article_text}
\"\"\""""

ARTICLE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "number"},
        "sentiment": {"type": "number"},
        "catalysts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "label": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["category", "label", "summary"],
            },
        },
        "price_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"event": {"type": "string"}, "price": {"type": "string"}},
                "required": ["event", "price"],
            },
        },
    },
    "required": ["relevance", "sentiment", "catalysts", "price_claims"],
}

# ── Event attribution (one call per price-event day) ──
EVENT_ATTRIBUTION_PROMPT = """Explain what drove {ticker}'s stock on {date}.

Market data for that session:
{event_data}

News stories published in the {days_before} trading days before and on that session (most relevant first):
{stories}

Using only these stories, explain the most likely driver(s) of the move in 1-3 sentences, citing story ids like [S12].
- The driver must fit the direction: good news does not explain a drop, and bad news does not explain a rise.
- Use the move type computed from prices: call a move "market-wide" only when it says market-wide. A stock-specific move needs a company-specific cause.
- Many moves have no clear news cause. If no story plausibly explains this one in its direction, use "unexplained" and say so. Do not stretch weak or contradictory stories into a cause.

Return JSON with "explanation", "primary_driver" (one catalyst category from: {categories}; or "market-wide"; or "unexplained"), "cited_stories" (list of story ids you relied on), and "confidence" (0.0-1.0)."""

EVENT_ATTRIBUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "primary_driver": {"type": "string"},
        "cited_stories": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["explanation", "primary_driver", "cited_stories", "confidence"],
}

# ── Report (one call per ticker × time range) ──
REPORT_PROMPT = """Write an investor briefing on {company} ({ticker}) for {start_date} to {end_date} ({time_range}). Today is {as_of}.

All figures below were computed from exchange prices and {article_count} news articles ({story_count} distinct stories). Treat them as fact and do not compute new numbers.

PRICE PERFORMANCE
{performance}

KEY PRICE EVENTS AND WHAT THE NEWS SAYS DROVE THEM (most significant first)
{events}

NEWS THEMES (share of relevant articles by catalyst, with average sentiment)
{themes}

SENTIMENT TREND
{sentiment_trend}

MOST COVERED STORIES
{top_stories}

Write these sections in Markdown:
1. **Summary**: 2-3 sentences on how the stock did and why.
2. **What moved the stock**: the most important events with their drivers and dates.
3. **Themes to watch**: the dominant news themes and what they imply.
4. **Sentiment**: the mood and how it changed.
5. **Takeaway**: 1-2 sentences for an investor.

Use only the information above, cite dates, and say so when a move has no clear news explanation."""

# ── RAG ──
RAG_QA_PROMPT = """You are a financial Q&A assistant. Today is {as_of}. Answer the question using ONLY the context below, which contains dated news articles and dated explanations of price moves.

Context:
{context}

Question: {question}

Give a clear, concise answer grounded in the context. Mention dates, prefer the most recent information, and say how old it is using the "published ... ago" note on each item (don't compute ages yourself). If the context doesn't answer the question, say so."""

# ── Judges ──
REPORT_JUDGE_PROMPT = """You are an evaluation judge. Rate this AI-generated investor briefing against the exact data it was written from.

Source data given to the writer:
\"\"\"
{source_text}
\"\"\"

Briefing:
\"\"\"
{summary_text}
\"\"\"

Rate each dimension 1-5 with a brief justification. Return JSON:
{{
  "hallucination": {{"score": <1=many claims not in the source data, 5=every claim is supported>, "justification": "<brief>"}},
  "faithfulness": {{"score": <1=misrepresents the source data, 5=perfectly faithful>, "justification": "<brief>"}},
  "relevancy": {{"score": <1=irrelevant to an investor, 5=highly relevant>, "justification": "<brief>"}}
}}"""

ATTRIBUTION_JUDGE_PROMPT = """You are an evaluation judge. An analyst explained a stock move using the news stories below.

Market data:
{event_data}

Stories available to the analyst:
{stories}

Explanation:
\"\"\"
{explanation}
\"\"\"

Rate from 1-5. Return JSON:
{{
  "groundedness": {{"score": <1=claims not supported by the cited stories, 5=fully supported>, "justification": "<brief>"}},
  "plausibility": {{"score": <1=the stories could not have caused this move, 5=a convincing cause>, "justification": "<brief>"}}
}}"""

RAG_JUDGE_PROMPT = """You are an evaluation judge for a question-answering system.

Question: {question}

Retrieved context:
\"\"\"
{context}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Rate from 1-5. Return JSON:
{{
  "context_relevance": {{"score": <1=retrieved context is unrelated to the question, 5=exactly what is needed>, "justification": "<brief>"}},
  "answer_relevance": {{"score": <1=does not address the question, 5=directly answers it>, "justification": "<brief>"}},
  "groundedness": {{"score": <1=answer is not supported by the context, 5=fully supported>, "justification": "<brief>"}}
}}"""
