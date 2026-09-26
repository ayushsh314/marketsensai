import numpy as np
import pandas as pd

from marketsensai.agents import parse_price, validate_analysis
from marketsensai.cache import JsonlCache
from marketsensai.llm import generate_json, parse_json_response
from marketsensai.stories import cluster_stories


def test_parse_json_direct_fenced_embedded_and_garbage():
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Sure! Here it is: {"a": 1} Hope that helps.') == {"a": 1}
    assert parse_json_response("no json here") == {}
    assert parse_json_response("[1, 2]") == {}


def test_generate_json_retries_unparseable_and_passes_schema(fake_llm):
    prompt = 'Analyze this news item about Apple (AAPL), published x.\n\n"""\ngarbled launch\n"""'
    [parsed] = generate_json(fake_llm, [prompt], 64, schema={"type": "object"})
    assert parsed["catalysts"][0]["category"] == "product launch"
    assert len(fake_llm.calls) == 2  # original + one retry
    assert fake_llm.schemas == [{"type": "object"}] * 2


def test_parse_price():
    assert parse_price("$1,195.50 per share") == 1195.5
    assert parse_price("186.61-190") == 186.61
    assert parse_price("") is None


def test_validate_analysis_normalizes_and_clamps():
    out = validate_analysis({"relevance": 3, "sentiment": "-2", "catalysts": [
        {"category": "Lawsuit", "label": "DOJ suit", "summary": "s"}, "junk"],
        "price_claims": [{"event": "record high", "price": "$200"}, {"event": ""}]})
    assert out["relevance"] == 1.0 and out["sentiment"] == -1.0
    assert out["catalysts"] == [{"category": "legal_regulatory", "label": "DOJ suit", "summary": "s"}]
    assert out["price_claims"] == [{"event": "record high", "label": "All-Time High", "price": 200.0}]
    assert validate_analysis({})["parse_failed"] is True


def test_jsonl_cache_resumes_and_respects_namespace(tmp_path):
    path = str(tmp_path / "c.jsonl")
    JsonlCache(path, "m1").put_many([("a", {"x": 1}), ("b", {"x": 2})])
    with open(path, "a") as f:
        f.write('{"ns": "m1", "key": "c", "val')  # torn write from an interrupted run
    resumed = JsonlCache(path, "m1")
    assert len(resumed) == 2 and resumed.get("b") == {"x": 2}
    assert len(JsonlCache(path, "m2")) == 0  # new model/prompt version → nothing reused


def test_cluster_stories_links_similar_articles_within_window():
    df = pd.DataFrame({
        "ticker": "AAPL", "article_id": list("abcd"), "title": ["t1", "t2", "t3", "t4"],
        "publisher": ["p1", "p2", "p1", "p3"],
        "trading_date": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-20", "2026-09-02"]),
    })
    same = np.array([1.0, 0.0])
    emb = np.stack([same, same, same, np.array([0.0, 1.0])])
    story_of, stories = cluster_stories(df, emb, threshold=0.9, window_days=2)
    assert story_of.iloc[0] == story_of.iloc[1]
    assert story_of.iloc[2] != story_of.iloc[0]  # same text but 19 days later → a different story
    assert story_of.iloc[3] != story_of.iloc[1]  # same day, different content
    big = stories.set_index("story_id").loc[story_of.iloc[0]]
    assert big["n_articles"] == 2 and big["n_publishers"] == 2
