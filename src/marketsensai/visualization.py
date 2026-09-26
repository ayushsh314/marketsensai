"""Poster-ready Plotly visualizations. Figures are saved as HTML, and as PNG when kaleido works.

Colors follow a validated categorical order (slot order is the colorblind-safety mechanism);
tickers keep the same color in every chart. Slots 3-5 are below 3:1 contrast on the light
surface, so charts that use them carry direct labels, hover text, or a CSV alongside.
"""

import os
import textwrap
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import TIME_RANGE_DELTAS, range_bounds
from .evaluation import daily_sentiment, spearman
from .prices import MarketData, between

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = {"positive": "#2a78d6", "negative": "#e34948", "neutral": "#898781"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def ticker_colors(tickers: List[str]) -> Dict[str, str]:
    """Fixed slot per ticker; past eight tickers the rest fold into muted gray rather than invent hues."""
    return {t: CATEGORICAL[i] if i < len(CATEGORICAL) else MUTED for i, t in enumerate(tickers)}


def _style(fig, title: str, **layout):
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=INK)),
        font=dict(family=FONT, size=14, color=INK_2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        barcornerradius=4, bargap=0.3, bargroupgap=0.08,
        uniformtext=dict(minsize=12, mode="show"),  # same label size on every bar
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        **layout,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


def _save(fig, viz_dir: str, name: str, png: bool) -> None:
    fig.write_html(os.path.join(viz_dir, f"{name}.html"))
    if png:
        try:
            fig.write_image(os.path.join(viz_dir, f"{name}.png"), scale=3)
        except Exception as e:  # kaleido is often broken on Colab / headless boxes
            print(f"   ⚠️ PNG export failed for {name}: {e}")
    print(f"   ✅ {name} saved")


def _wrap(text: str, width: int = 70) -> str:
    return "<br>".join(textwrap.wrap(str(text), width))


def price_timeline(ticker: str, market: MarketData, attributions: Dict[str, Dict], start: datetime, end: datetime):
    """Close price with every event day; hover shows the events and what the news says drove them."""
    window = between(market.prices[ticker], start, end, col=None)
    fig = go.Figure(go.Scatter(
        x=window.index, y=window["Close"], mode="lines", name="Close", line=dict(color=CATEGORICAL[0], width=2),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
    ))
    days = between(market.event_days[ticker], start, end)
    groups = [("Up event", "up", "triangle-up", CATEGORICAL[1]), ("Down event", "down", "triangle-down", CATEGORICAL[2]),
              ("Other event", "neutral", "circle", CATEGORICAL[3])]
    for name, direction, symbol, color in groups:
        g = days[days["direction"] == direction]
        if g.empty:
            continue
        hover = []
        for row in g.itertuples():
            a = attributions.get(str(row.date.date()), {})
            hover.append(f"<b>{', '.join(row.labels)}</b><br>Driver: {a.get('primary_driver', 'not analyzed')}"
                         f"<br>{_wrap(a.get('explanation', ''))}")
        fig.add_trace(go.Scatter(
            x=g["date"], y=g["close"], mode="markers", name=name, customdata=hover,
            marker=dict(symbol=symbol, size=np.clip(6 + g["significance"].astype(float) * 1.2, 8, 22),
                        color=color, line=dict(color=SURFACE, width=2)),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<br>%{customdata}<extra></extra>",
        ))
    return _style(fig, f"{ticker}: Price Events and What Drove Them (hover for explanations; size = significance)",
                  yaxis_title="Close (USD)", width=1150, height=520, hovermode="closest")


def sentiment_vs_returns(enriched: pd.DataFrame, market: MarketData, tickers: List[str]):
    colors = ticker_colors(tickers)
    fig = make_subplots(rows=1, cols=len(tickers), shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[" "] * len(tickers))
    for i, t in enumerate(tickers, start=1):
        daily = daily_sentiment(enriched[enriched["ticker"] == t]).join(market.features[t][["abn_ret"]], how="inner")
        rho = spearman(daily["sentiment"], daily["abn_ret"]) if not daily.empty else None
        fig.layout.annotations[i - 1].text = f"{t}  ρ = {rho if rho is not None else 'n/a'}"
        fig.add_trace(go.Scatter(
            x=daily["sentiment"], y=daily["abn_ret"] * 100, mode="markers", name=t, showlegend=False,
            customdata=np.stack([daily.index.strftime("%Y-%m-%d"), daily["articles"]], axis=-1) if len(daily) else None,
            marker=dict(size=8, color=colors[t], opacity=0.7, line=dict(color=SURFACE, width=1)),
            hovertemplate=f"{t} %{{customdata[0]}}<br>sentiment %{{x:+.2f}} (%{{customdata[1]}} articles)"
                          "<br>abnormal return %{y:+.2f}%<extra></extra>",
        ), row=1, col=i)
    fig.update_xaxes(title_text="Daily news sentiment", range=[-1, 1])
    fig.update_yaxes(title_text="Abnormal return (%)", col=1)
    return _style(fig, "Daily News Sentiment vs. Same-Day Abnormal Return (Spearman ρ)",
                  width=max(650, 360 * len(tickers)), height=460)


def sentiment_trend(enriched: pd.DataFrame, tickers: List[str]):
    colors = ticker_colors(tickers)
    fig = go.Figure()
    for t in tickers:
        daily = daily_sentiment(enriched[enriched["ticker"] == t])
        if daily.empty:
            continue
        weekly = daily.assign(w=daily["sentiment"] * daily["articles"]).resample("W").sum(numeric_only=True)
        weekly = weekly[weekly["articles"] > 0]
        fig.add_trace(go.Scatter(
            x=weekly.index, y=weekly["w"] / weekly["articles"], mode="lines+markers", name=t,
            line=dict(color=colors[t], width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
            customdata=weekly["articles"], hovertemplate=f"{t} week of %{{x|%Y-%m-%d}}: %{{y:+.2f}} "
                                                         "(%{customdata} articles)<extra></extra>",
        ))
    return _style(fig, "Weekly News Sentiment by Ticker", yaxis_title="Sentiment (−1 bearish, +1 bullish)",
                  yaxis_range=[-1, 1], width=1000, height=450)


def catalyst_mix(enriched: pd.DataFrame, tickers: List[str]):
    rel = enriched[(~enriched["parse_failed"]) & (enriched["relevance"] >= 0.5)]
    rows = rel.explode("catalysts").dropna(subset=["catalysts"])
    share = rows.groupby(["ticker", "catalysts"]).size().unstack(fill_value=0)
    share = share.div(rel.groupby("ticker").size(), axis=0).reindex([t for t in tickers if t in share.index])
    share = share[share.sum().sort_values(ascending=False).index]
    fig = go.Figure(go.Heatmap(
        z=share.values * 100, x=list(share.columns), y=list(share.index), xgap=2, ygap=2, zmin=0,
        colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
        text=np.round(share.values * 100).astype(int), texttemplate="%{text}%", colorbar=dict(title="% articles"),
        hovertemplate="%{y} · %{x}: %{z:.1f}% of relevant articles<extra></extra>",
    ))
    return _style(fig, "What the News Is About: Share of Relevant Articles by Catalyst",
                  width=1100, height=160 + 70 * len(share))


def catalyst_impact(impact: List[Dict]):
    cats = [r["category"] for r in impact]
    series = [("Bullish-toned days", "bullish_days_mean_abn_ret_pct", DIVERGING["positive"]),
              ("Bearish-toned days", "bearish_days_mean_abn_ret_pct", DIVERGING["negative"])]
    fig = go.Figure([
        go.Bar(name=name, x=cats, y=[r[key] for r in impact], marker_color=color,
               text=["–" if r[key] is None else f"{r[key]:+.2f}%" for r in impact], textposition="outside",
               textfont=dict(color=INK_2), customdata=[r["days"] for r in impact], cliponaxis=False,
               hovertemplate=f"%{{x}} · {name}: mean abnormal return %{{y:+.2f}}%<br>%{{customdata}} catalyst-days"
                             "<extra></extra>")
        for name, key, color in series
    ])
    vals = [abs(r[k]) for r in impact for _, k, _ in series if r[k] is not None] or [1]
    pad = max(vals) * 1.3
    return _style(fig, "Mean Same-Day Abnormal Return on Days a Catalyst Was Heavily Covered (≥3 articles), by Tone",
                  barmode="group", yaxis_title="Abnormal return (%)", yaxis_range=[-pad, pad],
                  width=max(900, 80 * len(cats)), height=480)


def event_explanations(per_ticker: Dict[str, Dict]):
    tickers = list(per_ticker)
    series = [("Had news in the window", "with_news_in_window", CATEGORICAL[0]),
              ("Explained by the news", "explained", CATEGORICAL[1])]
    fig = go.Figure([
        go.Bar(name=name, x=tickers, y=[per_ticker[t]["event_explanations"].get(key) for t in tickers],
               marker_color=color, textposition="outside", textfont=dict(color=INK_2),
               text=[f"{v:.0%}" if v is not None else "–"
                     for v in (per_ticker[t]["event_explanations"].get(key) for t in tickers)],
               hovertemplate=f"%{{x}} · {name}: %{{y:.0%}}<extra></extra>")
        for name, key, color in series
    ])
    return _style(fig, "Share of Price-Event Days the News Can Explain", barmode="group",
                  yaxis_range=[0, 1.15], yaxis_tickformat=".0%", width=800, height=430)


def judge_scores(evaluation: Dict):
    s = evaluation["summary"]
    groups = [("Reports", ["report_hallucination", "report_faithfulness", "report_relevancy"]),
              ("Event explanations", ["explanation_groundedness", "explanation_plausibility"]),
              ("RAG answers", ["rag_context_relevance", "rag_answer_relevance", "rag_groundedness"])]
    fig = go.Figure()
    for i, (name, keys) in enumerate(groups):
        keys = [k for k in keys if s.get(k) is not None]
        if not keys:
            continue
        labels = [k.split("_", 1)[1].replace("_", " ") for k in keys]
        fig.add_trace(go.Bar(
            name=name, y=[f"{name}: {l}" for l in labels], x=[s[k] for k in keys], orientation="h",
            marker_color=CATEGORICAL[i], text=[f"{s[k]:.1f}" for k in keys], textposition="outside",
            textfont=dict(color=INK_2), cliponaxis=False, hovertemplate="%{y}: %{x:.2f} / 5<extra></extra>",
        ))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "LLM-as-Judge Quality Scores (5 = best)", xaxis_range=[0, 5.5], xaxis_title="Score (1–5)",
                  width=950, height=460)


def time_to_insight(tp: Dict):
    minutes = [tp["manual_estimate_seconds"] / 60, tp["pipeline_seconds"] / 60]
    labels = [f"Manual reading (assumed {tp['manual_minutes_per_article_assumed']} min/article)", "MarketSensAI"]
    fig = go.Figure(go.Bar(
        y=labels, x=minutes, orientation="h", marker_color=[MUTED, CATEGORICAL[0]],
        text=[f"{m:,.1f} min" for m in minutes], textposition="outside", textfont=dict(color=INK),
        hovertemplate="%{y}: %{x:,.1f} min<extra></extra>",
    ))
    speedup = f" — {tp['speedup_factor']}× faster" if tp.get("speedup_factor") else ""
    return _style(fig, f"Time-to-Insight for {tp['articles']:,} Articles{speedup}", xaxis_title="Minutes",
                  width=850, height=300, showlegend=False, xaxis_range=[0, max(minutes) * 1.25])


def make_all(results: Dict[str, Dict], evaluation: Dict, enriched: pd.DataFrame, market: MarketData,
             as_of: datetime, time_ranges: List[str], viz_dir: str, png: bool = False) -> None:
    print("🎨 Generating visualizations...\n")
    os.makedirs(viz_dir, exist_ok=True)
    for name in os.listdir(viz_dir):  # this directory only ever holds generated charts and tables
        if name.endswith((".html", ".png", ".csv")):
            os.remove(os.path.join(viz_dir, name))
    tickers = list(results)
    start, end = range_bounds(max(time_ranges, key=TIME_RANGE_DELTAS.get), as_of)
    for t in tickers:
        _save(price_timeline(t, market, results[t]["attributions"], start, end), viz_dir, f"price_events_{t}", png)
        # Table view of every event day and its explanation.
        pd.DataFrame(sorted(results[t]["attributions"].values(), key=lambda a: a["date"])).drop(
            columns=["stories_text"], errors="ignore").to_csv(os.path.join(viz_dir, f"event_explanations_{t}.csv"),
                                                              index=False)
    if not enriched.empty:
        _save(sentiment_vs_returns(enriched, market, tickers), viz_dir, "sentiment_vs_returns", png)
        _save(sentiment_trend(enriched, tickers), viz_dir, "sentiment_trend", png)
        _save(catalyst_mix(enriched, tickers), viz_dir, "catalyst_mix", png)
    if evaluation:
        if evaluation.get("catalyst_impact"):
            _save(catalyst_impact(evaluation["catalyst_impact"]), viz_dir, "catalyst_impact", png)
        _save(event_explanations(evaluation["per_ticker"]), viz_dir, "event_explanations", png)
        _save(judge_scores(evaluation), viz_dir, "judge_scores", png)
        if evaluation["throughput"].get("speedup_factor"):  # only for runs that analyzed every article
            _save(time_to_insight(evaluation["throughput"]), viz_dir, "time_to_insight", png)
