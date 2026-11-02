# src/api.py
# Phase 8 — Retrieval API.
#
# Serves hybrid + rerank over HTTP, behind the same score gate as Phase 7. It
# returns handbook passages, not a generated answer, so it needs no API key.
# `confident` only means the top passage cleared the gate; near-miss questions can
# pass it, so it doesn't prove the passages answer the question.
#
# Provides:
#   GET  /health -> {"status": "ok"}
#   POST /query  -> {"question", "confident", "top_score", "passages"}
#
# Usage:
#   uvicorn src.api:app --port 8000
#   Interactive docs: http://127.0.0.1:8000/docs

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints

from src.rerank import DEFAULT_DEPTH, MIN_RELEVANCE_SCORE, search_reranked

DEFAULT_K = 5
MAX_QUESTION_CHARS = 1000

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
]


class QueryRequest(BaseModel):
    question: Question
    # The reranker only sees DEFAULT_DEPTH candidates, so k can't exceed it.
    k: int = Field(default=DEFAULT_K, ge=1, le=DEFAULT_DEPTH)


class Passage(BaseModel):
    rank: int
    score: float
    chunk_id: str
    source_path: str
    doc_title: str
    section_heading: str
    text: str


class QueryResponse(BaseModel):
    question: str
    confident: bool
    top_score: float | None
    passages: list[Passage]


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the indexes and models before accepting requests, so the first request
    # isn't slow and worker threads don't race to load them.
    search_reranked("warm up", k=1)
    yield


app = FastAPI(title="Hybrid Search RAG", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# A plain def: FastAPI runs it in a worker thread, so CPU-bound retrieval doesn't
# block the event loop.
@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    hits = search_reranked(request.question, k=request.k)
    top_score = hits[0]["score"] if hits else None
    confident = top_score is not None and top_score >= MIN_RELEVANCE_SCORE
    passages = [
        Passage(**hit, source_path=hit["chunk_id"].split("::")[0]) for hit in hits
    ] if confident else []
    return QueryResponse(
        question=request.question,
        confident=confident,
        top_score=top_score,
        passages=passages,
    )
