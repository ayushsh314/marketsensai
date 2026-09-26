from datetime import datetime

import pandas as pd

from marketsensai.data import (
    assign_trading_dates,
    extract_milestone_hint,
    load_corpus,
    parse_kaggle_row,
    save_corpus,
    select_articles,
)


def test_milestone_hint_from_title():
    assert extract_milestone_hint("Apple hits all-time high") == "All-Time High"
    assert extract_milestone_hint("MSFT reaches 52-week low") == "52-Week Low"
    assert extract_milestone_hint("Amazon at 4-week high") == "4-Week High"
    assert extract_milestone_hint("Amazon at 70-week high") == "16-Month High"
    assert extract_milestone_hint("Apple at 12-month high") == "52-Week High"
    assert extract_milestone_hint("Apple breakout confirmed") == "Breakout Event"
    assert extract_milestone_hint("Apple releases new iPhone") is None


def test_milestone_hint_no_bucket_for_odd_horizons():
    # A 13-week high used to be mislabeled as a 16-month high.
    assert extract_milestone_hint("Apple at 13-week high") is None


def test_parse_kaggle_row_extracts_fields():
    row = {
        "Title": "Apple hits all-time high",
        "Content": "Apple hits all-time highUnited States Apple stock reached a high of 186.612023-06-16T13:30:03.193",
        "Tag": "",
        "Category": "Stocks",
    }
    parsed = parse_kaggle_row(row)
    assert parsed["tickers"] == ["AAPL"]
    assert parsed["date"] == "2023-06-16"
    assert parsed["price"] == 186.61
    assert parsed["milestone_hint"] == "All-Time High"
    assert "2023-06-16T" not in parsed["content_clean"]


def test_parse_kaggle_row_relative_date_and_multiple_companies():
    row = {"Title": "Microsoft and Apple update", "Content": "Microsoft and Apple shares moved 18 days ago",
           "Tag": "", "Category": ""}
    parsed = parse_kaggle_row(row)
    assert parsed["tickers"] == ["AAPL", "MSFT"]
    assert parsed["date"] == "2023-05-28"


def test_parse_kaggle_row_whole_words_only():
    row = {"Title": "Pineapple prices", "Content": "Pineapple harvest news", "Tag": "", "Category": ""}
    assert parse_kaggle_row(row)["tickers"] == []


def test_select_articles_respects_range_and_as_of():
    df = pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "article_id": ["a", "b", "c"],
        "parsed_date": [datetime(2023, 6, 10), datetime(2022, 1, 1), datetime(2023, 6, 10)],
    })
    as_of = datetime(2023, 6, 15)
    assert list(select_articles(df, "AAPL", "1M", as_of)["article_id"]) == ["a"]
    # Nothing in range → empty, never a silent fallback to every article
    assert select_articles(df, "AAPL", "1D", as_of).empty


def test_select_articles_takes_everything_by_default():
    dates = [datetime(2023, 6, 14) - pd.Timedelta(days=d) for d in range(300)]
    df = pd.DataFrame({"ticker": "AAPL", "article_id": [str(i) for i in range(300)], "parsed_date": dates})
    assert len(select_articles(df, "AAPL", "1Y", datetime(2023, 6, 15))) == 300


def test_select_articles_spreads_over_time():
    # 50 articles on one busy day plus one per week for 10 weeks; a cap of 10 must not all be the busy day.
    busy = [datetime(2023, 6, 14)] * 50
    weekly = [datetime(2023, 6, 14) - pd.Timedelta(weeks=w) for w in range(1, 11)]
    dates = busy + weekly
    df = pd.DataFrame({"ticker": "AAPL", "article_id": [str(i) for i in range(len(dates))], "parsed_date": dates})
    picked = select_articles(df, "AAPL", "6M", datetime(2023, 6, 15), 10)
    assert len(picked) == 10
    assert picked["parsed_date"].nunique() >= 8


def test_corpus_roundtrip(tmp_path, articles):
    path = tmp_path / "articles.json"
    save_corpus(articles, datetime(2026, 9, 25), str(path))
    loaded, as_of = load_corpus(str(path))
    assert as_of == datetime(2026, 9, 25)
    assert list(loaded["article_id"]) == list(articles["article_id"])
    assert loaded["parsed_date"].iloc[0] == pd.Timestamp("2026-09-14")
    assert loaded["published_at"].iloc[0] == articles["published_at"].iloc[0]


def test_assign_trading_dates_uses_the_market_close():
    days = pd.bdate_range("2026-09-10", "2026-09-21")  # Thu 10th … Mon 21st
    df = pd.DataFrame({
        "published_at": ["2026-09-14T15:00:00+00:00",   # Mon 11am ET → Monday's session
                         "2026-09-14T21:00:00+00:00",   # Mon 5pm ET → after close → Tuesday
                         "2026-09-12T15:00:00+00:00",   # Saturday → Monday
                         None],                         # date only → that day's session
        "parsed_date": pd.to_datetime(["2026-09-14", "2026-09-14", "2026-09-12", "2026-09-11"]),
    })
    out = assign_trading_dates(df, days)
    assert [d.strftime("%a %d") for d in out] == ["Mon 14", "Tue 15", "Mon 14", "Fri 11"]
