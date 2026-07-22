# Hybrid Search RAG over Internal Docs

A retrieval-augmented generation system that answers natural-language questions over a
corpus of internal-style documentation using **hybrid retrieval** (BM25 keyword search +
dense vector search), **Reciprocal Rank Fusion**, **cross-encoder reranking**, and
**NLI-based citation verification**.

> Status: 🚧 in development. This README will hold the architecture diagram and the
> evaluation results table as the project progresses.

## Why hybrid?

Keyword search (BM25) nails exact terms — names, acronyms, error codes — but misses
paraphrases. Dense (embedding) search captures meaning but misses rare literal tokens.
Fusing both, then reranking the merged set with a cross-encoder, beats either alone.
The eval harness in `eval/` measures exactly how much.

## Planned architecture

```
documents ──> ingest (chunk + metadata) ──> BM25 index
                                         └─> dense index (Chroma)
                                                   │
query ──> BM25 + dense retrieval ──> RRF fusion ──> cross-encoder rerank
                                                   │
                                       top-k chunks ──> LLM answer + citations
                                                   │
                                       citation verification (NLI) ──> faithfulness score
```

## Corpus

A subset of the public [GitLab Handbook](https://gitlab.com/gitlab-com/content-sites/handbook)
(HR policies, security, engineering) — real internal company documentation that GitLab
publishes openly. Download script: see `scripts/` (added in Phase 0/1).

## Stack

- **Retrieval:** `rank-bm25` (sparse), `sentence-transformers` + `chromadb` (dense)
- **Reranking:** `bge-reranker-base` cross-encoder
- **Verification:** NLI model (`nli-deberta-v3-base`)
- **Generation:** Claude API
- **Serving:** FastAPI
- All models run **locally on CPU** by default.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -e .
```

## Build phases

- [x] Phase 0 — repo + skeleton
- [ ] Phase 1 — ingestion & chunking
- [ ] Phase 2 — BM25 retrieval
- [ ] Phase 3 — dense retrieval
- [ ] Phase 4 — hybrid fusion (RRF)
- [ ] Phase 5 — evaluation harness
- [ ] Phase 6 — cross-encoder reranking
- [ ] Phase 7 — generation with citations
- [ ] Phase 7.5 — citation verification
- [ ] Phase 8 — FastAPI + polish
