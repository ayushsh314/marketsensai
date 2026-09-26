import numpy as np
import pandas as pd

from marketsensai.prices import EventRules, _threshold_crossings, compute_features, detect_events, summarize_event_days
from marketsensai.taxonomy import normalize_catalyst, normalize_price_event

from conftest import JUMP_DAY, synthetic_market, synthetic_prices


def test_normalize_price_event():
    assert normalize_price_event("52-Week High") == "52-Week High"
    assert normalize_price_event("12-month high") == "52-Week High"
    assert normalize_price_event("record low") == "All-Time Low"
    assert normalize_price_event("three-month high") == "13-Week High"
    assert normalize_price_event("six-month low") == "26-Week Low"
    assert normalize_price_event("broke out above resistance") == "Breakout"
    assert normalize_price_event("entered a bear market") == "Bear Market"
    assert normalize_price_event("gapped up") == "Gap Up"
    assert normalize_price_event("fell 8%") is None
    assert normalize_price_event(None) is None


def test_normalize_catalyst_keeps_known_and_maps_synonyms():
    assert normalize_catalyst("earnings") == "earnings"
    assert normalize_catalyst("Legal/Regulatory") == "legal_regulatory"
    assert normalize_catalyst("Lawsuit") == "legal_regulatory"
    assert normalize_catalyst("analyst upgrade") == "analyst"
    assert normalize_catalyst("stock buyback") == "capital"
    assert normalize_catalyst("tariffs") == "macro"
    assert normalize_catalyst("celebrity endorsement") == "other"


def test_normalize_catalyst_falls_back_to_the_label():
    assert normalize_catalyst("other", "price target raise") == "analyst"
    assert normalize_catalyst("other", "market sentiment") == "market"
    assert normalize_catalyst("other", "AI competition") == "technology"
    assert normalize_catalyst("other", "50th anniversary") == "other"
    assert normalize_catalyst("earnings", "anything") == "earnings"  # a known category wins


def test_jump_day_events():
    m = synthetic_market(["AAPL"])
    labels = set(m.events["AAPL"].query("date == @JUMP_DAY")["label"])
    assert {"Large Gain", "Gap Up", "Volume Spike", "Breakout"} <= labels
    day = m.event_days["AAPL"].set_index("date").loc[JUMP_DAY]
    assert day["direction"] == "up"
    # Only the strongest rung of the high/low ladder is kept on the collapsed day.
    assert sum(label.endswith("High") for label in day["labels"]) == 1
    assert day["significance"] == m.event_days["AAPL"]["significance"].max()


def test_market_wide_move_is_not_a_stock_specific_gain():
    rules = EventRules()
    bench = synthetic_prices(seed=5, jump=True)
    # Tracks the market plus a little noise of its own, so the jump is market-wide, not stock-specific.
    noise = np.random.default_rng(1).normal(0, 0.002, len(bench))
    stock = bench.copy()
    stock["Close"] = 50 * np.cumprod(1 + bench["Close"].pct_change().fillna(0) + noise)
    stock["Open"], stock["High"], stock["Low"] = stock["Close"], stock["Close"] * 1.002, stock["Close"] * 0.998
    feats = compute_features(stock, bench, rules)
    labels = set(detect_events(stock, feats, rules).query("date == @JUMP_DAY")["label"])
    assert "Market-Wide Rally" in labels
    assert "Large Gain" not in labels


def test_drawdown_crossings_need_recovery_to_rearm():
    dd = pd.Series([0, -0.11, -0.09, -0.12, -0.03, -0.12])
    # Fires at -11%; the wobble back to -9% and down to -12% doesn't re-fire; recovery to -3% re-arms.
    assert list(_threshold_crossings(dd, 0.10)) == [False, True, False, False, False, True]


def test_summarize_event_days_empty():
    assert summarize_event_days(pd.DataFrame()).empty
