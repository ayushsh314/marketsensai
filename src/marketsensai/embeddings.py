"""Article embeddings (MiniLM), computed once and cached; shared by story clustering and RAG."""

import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd

DOC_CHARS = 1000  # cap per document for embedding


def load_embed_model(model_name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📐 Loading embedding model: {model_name} ({device})")
    return SentenceTransformer(model_name, device=device)


def article_keys(df: pd.DataFrame) -> List[str]:
    return (df["ticker"] + "|" + df["article_id"]).tolist()


def article_text(row) -> str:
    return f"[{row['ticker']}] {row['title']} | {row['date']}\n{row['content_clean']}"[:DOC_CHARS]


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def embed_texts(texts: List[str], embed_model) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    return normalize(embed_model.encode(texts, show_progress_bar=len(texts) > 1000, batch_size=128))


def embed_articles(df: pd.DataFrame, embed_model, model_name: str, cache_dir: Optional[str] = None) -> np.ndarray:
    """Unit-normalized embeddings aligned with df's rows; only articles not already cached are embedded."""
    keys = article_keys(df)
    cached = {}
    vec_file = os.path.join(cache_dir, "embeddings.npy") if cache_dir else None
    key_file = os.path.join(cache_dir, "embeddings.json") if cache_dir else None
    if vec_file and os.path.exists(vec_file) and os.path.exists(key_file):
        with open(key_file) as f:
            meta = json.load(f)
        if meta.get("model") == model_name:
            vecs = np.load(vec_file)
            cached = dict(zip(meta["keys"], vecs))

    missing = [i for i, k in enumerate(keys) if k not in cached]
    if missing:
        print(f"   Embedding {len(missing):,} articles ({len(keys) - len(missing):,} cached)...")
        new = embed_texts([article_text(df.iloc[i]) for i in missing], embed_model)
        for i, v in zip(missing, new):
            cached[keys[i]] = v
    out = np.stack([cached[k] for k in keys]) if keys else np.zeros((0, 384), dtype="float32")

    if vec_file and missing:
        os.makedirs(cache_dir, exist_ok=True)
        all_keys = list(cached)
        np.save(vec_file, np.stack([cached[k] for k in all_keys]))
        with open(key_file, "w") as f:
            json.dump({"model": model_name, "keys": all_keys}, f)
    return out
