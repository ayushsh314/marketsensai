"""Group articles that cover the same news into stories.

Forty outlets rewriting one Reuters piece is one story with forty sources, not forty independent
signals. Stories carry that salience into event attribution and reports.
"""

from typing import Tuple

import numpy as np
import pandas as pd

STORY_COLUMNS = ["story_id", "ticker", "trading_date", "last_date", "n_articles", "n_publishers",
                 "headline", "article_ids"]


def _find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def cluster_stories(df: pd.DataFrame, emb: np.ndarray, threshold: float,
                    window_days: int) -> Tuple[pd.Series, pd.DataFrame]:
    """Link articles whose embeddings are ≥ threshold similar and ≤ window_days apart (single linkage).

    df holds one ticker's articles (with trading_date) and emb its unit-normalized embeddings, row-aligned.
    Returns (story_id per article, one row per story).
    """
    if df.empty:
        return pd.Series(dtype=str), pd.DataFrame(columns=STORY_COLUMNS)
    ticker = df["ticker"].iloc[0]
    trading = pd.to_datetime(df["trading_date"])
    if "parsed_date" in df:
        trading = trading.fillna(pd.to_datetime(df["parsed_date"]))
    order = np.argsort(trading.values, kind="stable")
    dates = trading.values[order].astype("datetime64[D]").astype(np.int64)
    e = emb[order]
    n = len(order)
    parent = np.arange(n)
    hi = 0
    for i in range(n):
        hi = max(hi, i + 1)
        while hi < n and dates[hi] - dates[i] <= window_days:
            hi += 1
        if hi > i + 1:
            sims = e[i + 1:hi] @ e[i]
            for j in np.nonzero(sims >= threshold)[0] + i + 1:
                ri, rj = _find(parent, i), _find(parent, j)
                if ri != rj:
                    parent[rj] = ri
    roots = np.array([_find(parent, i) for i in range(n)])

    # Number stories chronologically.
    root_order = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    labels_sorted = np.array([f"{ticker}-S{root_order[r] + 1}" for r in roots])
    story_of = pd.Series(index=df.index[order], data=labels_sorted).reindex(df.index)

    rows = []
    for sid, members in story_of.groupby(story_of, sort=False):
        idx = members.index
        pos = [df.index.get_loc(i) for i in idx]
        centroid = emb[pos].mean(axis=0)
        central = idx[int(np.argmax(emb[pos] @ centroid))]
        group = df.loc[idx]
        rows.append({
            "story_id": sid,
            "ticker": ticker,
            "trading_date": group["trading_date"].min(),
            "last_date": group["trading_date"].max(),
            "n_articles": len(idx),
            "n_publishers": group["publisher"].replace("", np.nan).nunique() or 1,
            "headline": df.at[central, "title"],
            "article_ids": list(group["article_id"]),
        })
    stories = pd.DataFrame(rows, columns=STORY_COLUMNS).sort_values("trading_date").reset_index(drop=True)
    return story_of, stories
