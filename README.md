# Hybrid Search RAG over Internal Docs

A retrieval-augmented generation system that answers natural-language questions over a
corpus of internal-style documentation using **hybrid retrieval** (BM25 keyword search +
dense vector search), **Reciprocal Rank Fusion**, **cross-encoder reranking**, and
**NLI-based citation verification**.

> **Status:** Phases 0–5 complete. Eval harness live — see results table below.
> Cross-encoder reranking (Phase 6) is next.

## Eval results (Phase 5)

42-question gold set over the GitLab Handbook corpus (security + people-group + people-policies).
k=10 retrieval; gold chunk must appear in top-k to count as a hit.

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|:---:|:---:|:---:|:---:|
| BM25 | 35.7% | 57.1% | 71.4% | 0.458 |
| Dense (BGE) | 35.7% | 64.3% | 76.2% | 0.464 |
| **Hybrid (RRF)** | **40.5%** | **71.4%** | **76.2%** | **0.527** |

**Hybrid Recall@5 (71.4%) beats BM25 by +14.3 pp and Dense by +7.1 pp.**
MRR lifts from 0.458 → 0.527 (+15% relative) — hybrid consistently ranks the right
chunk higher, not just finds it more often.

Cross-encoder reranking (Phase 6) will push these numbers further.

## Why hybrid?

Keyword search (BM25) nails exact terms — names, acronyms, error codes, URL paths —
but misses paraphrases. Dense (embedding) search captures meaning but misses rare
literal tokens. Fusing both, then reranking the merged set with a cross-encoder,
beats either alone.

The eval table above is the evidence. The overlap analysis below illustrates *why*
the two retrievers are complementary — it's development intuition, not a measurement.

**Retriever overlap during development** (7 hand-picked development queries;
see `src/compare_retrievers.py --hybrid`):

| Query | BM25 ∩ Dense (top-5) | Observation |
|---|:---:|---|
| `anti-harassment policy` | 1/5 | Dense puts correct doc at rank 1; BM25 had it at rank 3 |
| `how do I take unpaid leave` | 0/5 | Total divergence — both find valid but different leave chunks |
| `gitlab-com/gl-security` | 0/5 | BM25 finds URL matches; dense finds security architecture docs |
| `expense reimbursement` | 4/5 | High agreement — easy query, hybrid barely changes ranking |
| `onboarding laptop linux` | 1/5 | Different Linux setup chunks each |
| `time off` | 1/5 | Different time-off chunks each |
| `who do I contact about workplace complaints` | 0/5 | Dense 5/5 relevant; BM25 0/5 |

3 of 7 development queries had zero overlap — a signal worth pursuing, confirmed
later by the eval harness.

## Architecture

```
data/raw/*.md (556 GitLab Handbook files)
       │
       ▼
   ingest (Phase 1)
       │  · heading-aware section splitting
       │  · token-budgeted chunking (300–500 tok)
       │  · never splits tables
       │  · two-pass tiny-section merge
       ▼
   data/processed/chunks.json  (2,622 chunks, ~89 % between 100–500 tokens)
       │
       ├──> BM25 index (Phase 2)                     ─┐
       │      · Snowball-stemmed conservative tokenizer│
       │      · title + heading + body indexed         │
       │      · data/processed/bm25_index.pkl (7.3 MB) │
       │                                                ├─> Phase 4: RRF fusion
       └──> Dense index (Phase 3)                     ─┤   (top-k merge)
              · BAAI/bge-small-en-v1.5 (384-dim)      │
              · Chroma, hnsw:space=cosine             │
              · .chroma/ (39 MB)                      │
                                                       │
                     Phase 6: cross-encoder rerank ◀───┘
                                │
                                ▼
                     Phase 7: Claude answer with citations
                                │
                                ▼
                     Phase 7.5: NLI citation verification
                                │
                                ▼
                     Phase 8: FastAPI /query endpoint
```

## Corpus

A subset of the public [GitLab Handbook](https://gitlab.com/gitlab-com/content-sites/handbook):
`security/`, `people-group/`, and `people-policies/` — real internal company
documentation that GitLab publishes openly. **556 Markdown files, 5.1 MB raw**,
producing **2,622 chunks** after ingestion.

The `engineering/` directory (1,209 files) was excluded to keep CPU embedding
under 5 minutes end to end.

## Stack

- **Chunking:** `python-frontmatter`, `tiktoken` (`cl100k_base`)
- **Sparse retrieval:** `rank-bm25` with an `nltk` Snowball stemmer
- **Dense retrieval:** `sentence-transformers` (`BAAI/bge-small-en-v1.5`), `chromadb`
- **Reranking (planned):** `bge-reranker-base` cross-encoder
- **Verification (planned):** NLI model (`nli-deberta-v3-base`)
- **Generation (planned):** Claude API
- **Serving (planned):** FastAPI

All local models run **on CPU by default**. No GPU required.

## Quick start

```bash
# 1. Set up the environment
python -m venv .venv
.venv\Scripts\activate      # Windows (use source .venv/bin/activate on Unix)
pip install -e .

# 2. Fetch the corpus (shallow clone, then copy the subset)
git clone --depth 1 https://gitlab.com/gitlab-com/content-sites/handbook.git _handbook_tmp
mkdir -p data/raw
cp -r _handbook_tmp/content/handbook/security data/raw/
cp -r _handbook_tmp/content/handbook/people-group data/raw/
cp -r _handbook_tmp/content/handbook/people-policies data/raw/
rm -rf _handbook_tmp

# 3. Build the pipeline (regenerates every artifact)
python -m src.ingest          --input data/raw --output data/processed/chunks.json
python -m src.index_bm25                       # → data/processed/bm25_index.pkl
python -m src.index_dense --build --verify     # → .chroma/  (takes ~3 min on CPU)

# 4. Try both retrievers, side by side
python -m src.retrieve_bm25    "how do I take unpaid leave" --k 5
python -m src.retrieve_dense   "how do I take unpaid leave" --k 5
python -m src.compare_retrievers "how do I take unpaid leave" --k 5
```

## Build phases

| Phase | Description | Status |
|---|---|:---:|
| 0 | Repo + skeleton, venv, corpus in place | ✅ |
| 1 | Ingestion & chunking (2,622 chunks; 89 % between 100–500 tok) | ✅ |
| 2 | BM25 retrieval (Snowball stemmer, index at 7.3 MB, ~14 s build) | ✅ |
| 3 | Dense retrieval (BGE + Chroma, 39 MB, ~3 min build) + comparison CLI | ✅ |
| 4 | Hybrid fusion (Reciprocal Rank Fusion) | ✅ |
| 5 | Evaluation harness (recall@5: BM25 57.1% / Dense 64.3% / Hybrid 71.4%) | ✅ |
| 6 | Cross-encoder reranking (`bge-reranker-base`) | ⬜ |
| 7 | Generation with forced citations (Claude API) | ⬜ |
| 7.5 | NLI-based citation verification (faithfulness score) | ⬜ |
| 8 | FastAPI `/query` endpoint + polish | ⬜ |

## Known caveats

- **19.2 % of chunks exceed the BGE 512-token limit** and have their tails silently
  truncated by the dense encoder. Only 22 chunks are severely affected (>50 % lost —
  all giant tables that Phase 1 correctly refused to split). Phase 5 evals confirmed
  one miss is directly caused by this (1,537-token CorpSec Engineering chunk). Fix
  is deferred until the query set grows large enough to measure the lift.
- **Some chunks contain raw HTML** (`<iframe>`, Font Awesome `<i>` tags) that isn't
  Hugo shortcode syntax and slipped past the cleaner. Roughly 5 % of chunks;
  BM25 tokenization treats it as noise. Same triage plan.

## Repo layout

```
src/
├── ingest.py              # Phase 1  — parse + chunk Markdown
├── index_bm25.py          # Phase 2  — BM25 index build + persist
├── retrieve_bm25.py       # Phase 2  — search_bm25() + CLI
├── index_dense.py         # Phase 3  — Chroma index build + persist
├── retrieve_dense.py      # Phase 3  — search_dense() + CLI
├── compare_retrievers.py  # Phase 3/4 — side-by-side retriever comparison
└── retrieve_hybrid.py     # Phase 4  — RRF fusion + CLI

eval/
├── generate_questions.py  # Phase 5  — gold Q&A set generator
├── questions.json         # Phase 5  — 42 gold question → chunk_id pairs
└── run_eval.py            # Phase 5  — recall@1/5/10 + MRR harness

data/
├── raw/                   # source Markdown (gitignored, redownloadable)
└── processed/
    ├── chunks.json        # Phase 1 output (gitignored)
    └── bm25_index.pkl     # Phase 2 output (gitignored)

.chroma/                   # Phase 3 output (gitignored, ~39 MB)
PROGRESS.md                # detailed live tracker (open questions, per-phase notes)
```
