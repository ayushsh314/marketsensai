"""Retrieval-augmented Q&A over the articles and the pipeline's event explanations.

Retrieval is exact cosine search over unit-normalized MiniLM embeddings, the same computation as a
FAISS IndexFlatIP. At tens of thousands of documents a numpy matrix product takes milliseconds;
past a few million you would switch to an approximate index or a vector database.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .embeddings import article_text, embed_texts
from .prompts import RAG_QA_PROMPT


class RAGQAModule:
    """Dense retrieval (exact cosine) filtered by ticker, answered by the LLM from dated context."""

    def __init__(self, documents: List[str], metadata: List[Dict], embeddings: np.ndarray, embed_model, llm,
                 as_of: datetime, top_k: int = 6, recency_weight: float = 0.0, recency_half_life_days: int = 90):
        self.documents = documents
        self.metadata = metadata
        self.embed_model = embed_model
        self.llm = llm
        self.as_of = as_of
        self.top_k = top_k
        self.embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        # Recent documents get a small boost (recency_weight today, halving every half-life), so "what's
        # happening now" questions don't surface year-old articles that happen to match better.
        dates = pd.to_datetime([m["date"] for m in metadata], errors="coerce")
        age = np.nan_to_num((pd.Timestamp(as_of) - dates).days.to_numpy(dtype=float), nan=3650.0)
        self.recency = recency_weight * 0.5 ** (np.clip(age, 0, None) / recency_half_life_days)
        ids: Dict[str, List[int]] = {}
        for i, m in enumerate(metadata):
            ids.setdefault(m["ticker"], []).append(i)
        self._ids_by_ticker = {t: np.array(v, dtype="int64") for t, v in ids.items()}

    @property
    def tickers(self) -> List[str]:
        return sorted(self._ids_by_ticker)

    def retrieve(self, query: str, ticker: Optional[str] = None) -> List[Dict]:
        if ticker:
            if ticker not in self._ids_by_ticker:
                return []
            candidates = self._ids_by_ticker[ticker]  # filter first, so a ticker always gets its own top-k
        else:
            candidates = np.arange(len(self.documents))
        similarity = self.embeddings[candidates] @ embed_texts([query], self.embed_model)[0]
        scores = similarity + self.recency[candidates]
        top = np.argsort(-scores)[:self.top_k]
        return [{"text": self.documents[candidates[i]], "score": float(scores[i]),
                 "similarity": float(similarity[i]), **self.metadata[candidates[i]]} for i in top]

    def _age(self, date: str) -> str:
        """Precomputed so the model never has to do date arithmetic."""
        d = pd.to_datetime(date, errors="coerce")
        if pd.isna(d):
            return "date unknown"
        days = (pd.Timestamp(self.as_of) - d).days
        if days <= 1:
            return "published today" if days <= 0 else "published 1 day ago"
        if days < 60:
            return f"published {days} days ago"
        return f"published about {round(days / 30.4)} months ago"

    def answer(self, question: str, ticker: Optional[str] = None) -> Dict:
        retrieved = self.retrieve(question, ticker)
        context = "\n\n---\n\n".join(f"({self._age(r['date'])})\n{r['text']}"
                                       for r in sorted(retrieved, key=lambda r: r["date"], reverse=True))
        if not retrieved:
            scope = f"{ticker} is" if ticker else "Nothing relevant is"
            answer = f"{scope} not in the corpus (covered tickers: {', '.join(self.tickers)})."
        else:
            answer = self.llm.generate(RAG_QA_PROMPT.format(
                as_of=self.as_of.strftime("%Y-%m-%d"), context=context, question=question), max_tokens=512)
        return {
            "question": question, "ticker": ticker, "answer": answer, "context": context,
            "sources": [{k: r[k] for k in ("kind", "title", "date", "url", "score")} for r in retrieved],
        }


def event_documents(explanations: Dict[str, List[Dict]]) -> Tuple[List[str], List[Dict]]:
    docs, meta = [], []
    for ticker, items in explanations.items():
        for e in items:
            if not e.get("explanation"):
                continue
            title = f"{ticker} price move on {e['date']}: {', '.join(e['labels'])}"
            docs.append(f"[{ticker}] {title} | {e['date']}\n{e['event_data']}\nWhy: {e['explanation']}")
            meta.append({"kind": "event", "ticker": ticker, "date": e["date"], "title": title, "url": ""})
    return docs, meta


def build_rag(df_articles: pd.DataFrame, article_embeddings: np.ndarray, explanations: Dict[str, List[Dict]],
              llm, embed_model, as_of: datetime, top_k: int, recency_weight: float = 0.0,
              recency_half_life_days: int = 90) -> RAGQAModule:
    print("🗄️ Building the RAG index (articles + event explanations)...")
    docs = [article_text(row) for _, row in df_articles.iterrows()]
    meta = [{"kind": "article", "ticker": r.ticker, "date": r.date, "title": r.title, "url": r.url}
            for r in df_articles.itertuples()]
    ev_docs, ev_meta = event_documents(explanations)
    ev_emb = embed_texts(ev_docs, embed_model) if ev_docs else np.zeros((0, article_embeddings.shape[1]), "float32")
    emb = np.vstack([article_embeddings, ev_emb]) if len(ev_emb) else article_embeddings
    print(f"   {len(docs):,} articles + {len(ev_docs):,} event explanations")
    return RAGQAModule(docs + ev_docs, meta + ev_meta, emb, embed_model, llm, as_of, top_k,
                       recency_weight, recency_half_life_days)


OUT_OF_CORPUS_TICKER = "ZZZZ"  # checks that questions about uncovered tickers are declined, not invented


def demo_questions(tickers: List[str]) -> List[Tuple[str, Optional[str]]]:
    questions = []
    for t in tickers:
        questions += [
            (f"What drove {t}'s biggest stock moves recently?", t),
            (f"What are the main news themes around {t} right now, and is the sentiment positive or negative?", t),
        ]
    questions.append((f"Why did {OUT_OF_CORPUS_TICKER} stock fall?", OUT_OF_CORPUS_TICKER))
    return questions
