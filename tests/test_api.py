# tests/test_api.py
# Phase 8 — API behaviour with retrieval faked, so these run fast and don't load
# any models. TestClient only runs startup when used as a context manager.

import pytest
from fastapi.testclient import TestClient

from src import api
from src.rerank import DEFAULT_DEPTH, MIN_RELEVANCE_SCORE


def _hit(rank, score):
    return {
        "rank": rank,
        "score": score,
        "fused_rank": rank + 1,
        "chunk_id": f"people-group/leave.md::{rank:04d}",
        "doc_title": "Leave",
        "section_heading": "Unpaid leave",
        "text": f"passage {rank}",
        "token_count": 100,
    }


@pytest.fixture
def search(monkeypatch):
    """Fake search_reranked; set `.hits` to choose what it returns."""

    class FakeSearch:
        hits = [_hit(1, 4.0), _hit(2, 1.0)]
        calls = []

        def __call__(self, question, k):
            self.calls.append((question, k))
            return self.hits

    fake = FakeSearch()
    fake.calls = []
    monkeypatch.setattr(api, "search_reranked", fake)
    return fake


client = TestClient(api.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_query_returns_passages_with_sources_in_rank_order(search):
    response = client.post("/query", json={"question": "unpaid leave?"})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "unpaid leave?"
    assert body["confident"] is True
    assert body["top_score"] == 4.0
    assert [p["rank"] for p in body["passages"]] == [1, 2]
    assert body["passages"][0] == {
        "rank": 1,
        "score": 4.0,
        "chunk_id": "people-group/leave.md::0001",
        "source_path": "people-group/leave.md",
        "doc_title": "Leave",
        "section_heading": "Unpaid leave",
        "text": "passage 1",
    }


def test_query_below_gate_returns_no_passages_but_keeps_the_score(search):
    search.hits = [_hit(1, MIN_RELEVANCE_SCORE - 0.01)]
    body = client.post("/query", json={"question": "best pizza?"}).json()
    assert body["confident"] is False
    assert body["top_score"] == MIN_RELEVANCE_SCORE - 0.01
    assert body["passages"] == []


def test_query_exactly_at_gate_is_confident(search):
    search.hits = [_hit(1, MIN_RELEVANCE_SCORE)]
    body = client.post("/query", json={"question": "q"}).json()
    assert body["confident"] is True
    assert len(body["passages"]) == 1


def test_query_with_no_hits_is_not_confident(search):
    search.hits = []
    body = client.post("/query", json={"question": "q"}).json()
    assert body == {"question": "q", "confident": False, "top_score": None, "passages": []}


def test_query_passes_k_and_defaults_to_five(search):
    client.post("/query", json={"question": "q"})
    client.post("/query", json={"question": "q", "k": DEFAULT_DEPTH})
    assert search.calls == [("q", 5), ("q", DEFAULT_DEPTH)]


def test_query_strips_the_question(search):
    body = client.post("/query", json={"question": "  q \n"}).json()
    assert body["question"] == "q"
    assert search.calls == [("q", 5)]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * (api.MAX_QUESTION_CHARS + 1)},
        {"question": "q", "k": 0},
        {"question": "q", "k": DEFAULT_DEPTH + 1},
        {"question": "q", "k": "many"},
    ],
)
def test_query_rejects_bad_input_without_searching(search, payload):
    assert client.post("/query", json=payload).status_code == 422
    assert search.calls == []


def test_startup_loads_retrieval_before_serving(search):
    with TestClient(api.app) as started:
        assert len(search.calls) == 1
        assert started.get("/health").status_code == 200
