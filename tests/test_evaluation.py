import numpy as np
import pandas as pd

from marketsensai.evaluation import flip_label, number_check, rank_correlation
from marketsensai.taxonomy import EVENT_TYPES


def test_flip_label_mirrors_every_directional_event():
    for label, meta in EVENT_TYPES.items():
        flipped = flip_label(label)
        if meta.category == "drawdown":
            assert flipped is None
        elif meta.direction == "neutral":
            assert flipped == label
        else:
            assert flipped in EVENT_TYPES and EVENT_TYPES[flipped].direction != meta.direction
            assert flip_label(flipped) == label


def test_rank_correlation_reports_uncertainty():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=400))
    strong = rank_correlation(x, x + pd.Series(rng.normal(size=400)))
    assert strong["n"] == 400 and strong["p_value"] < 1e-6 and strong["ci95"][0] > 0.5
    noise = rank_correlation(x, pd.Series(rng.normal(size=400)))
    assert noise["ci95"][0] < 0 < noise["ci95"][1] and noise["p_value"] > 0.05
    assert rank_correlation(x[:3], x[:3]) == {"rho": None, "n": 3}


SOURCE = """PERIOD:
2026-08-28 to 2026-09-25 (1M), as of 2026-09-26; 450 articles, 300 stories

PERFORMANCE:
Return over the whole period: +6.30% (close 485.55 on 2026-08-28, the first session, to close 516.17 on 2026-09-25, the last session). This is not a one-day move.
SPY return over the same period: +0.26% (only its return is given, not its price); the stock outperformed SPY by 6.04 percentage points."""


def test_number_check_accepts_rounded_source_figures():
    report = ("MSFT rose 6.3% to $516.17 between 2026-08-28 and 2026-09-25, beating SPY by about 6 points "
              "across 450 articles. Its 52-week trend and the S&P 500 are noted. **1. Summary**")
    check = number_check(report, SOURCE)
    assert check["passed"], check
    assert check["numbers"] == 3  # 6.3, 516.17, 450; small counts, 52 and 500 are skipped


def test_number_check_flags_invented_figures_and_dates():
    report = "MSFT closed at $520.00 on 2026-09-24, while SPY traded at $222.05. Down 6.30% overall."
    check = number_check(report, SOURCE)
    assert not check["passed"]
    assert check["unsupported_numbers"] == ["520.00", "222.05"]
    assert check["unsupported_dates"] == ["2026-09-24"]
