# Hybrid Search RAG over Internal Docs

A retrieval system for natural-language questions over a corpus of internal company
documentation. It combines **BM25 keyword search** and **dense vector search** with
**Reciprocal Rank Fusion**, reorders the fused candidates with a **cross-encoder
reranker**, and is measured by a gold-set **evaluation harness** (Recall@k, MRR).
A local **NLI checker** tests whether a passage supports a statement, and a **FastAPI
service** serves the results over HTTP.

> **Status:** Retrieval-only. Phases 0–6, 7.5 and 8 are complete. An optional Claude answer
> generator (Phase 7) is included but has never completed a live API call, so it is tested
> only against a simulated API and answer quality is unmeasured.

## Eval results

42-question gold set over the GitLab Handbook corpus (security + people-group + people-policies).
k=10 retrieval; gold chunk must appear in top-k to count as a hit.

Latency is per query on CPU, models already loaded.

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Latency |
|---|:---:|:---:|:---:|:---:|:---:|
| BM25 | 35.7% | 57.1% | 71.4% | 0.458 | 46 ms |
| Dense (BGE) | 35.7% | 64.3% | 76.2% | 0.464 | 20 ms |
| Hybrid (RRF) | 40.5% | 71.4% | 76.2% | 0.527 | 68 ms |
| **Hybrid + rerank** (MiniLM-L-6) | **50.0%** | **76.2%** | **83.3%** | **0.599** | 0.9 s |
| Hybrid + rerank (`bge-reranker-base`) | 50.0% | 69.0% | 81.0% | 0.578 | 6.1 s |

**Hybrid Recall@5 (71.4%) beats BM25 by +14.3 pp and Dense by +7.1 pp.**
MRR lifts from 0.458 → 0.527 (+15% relative) — hybrid consistently ranks the right
chunk higher, not just finds it more often.

**Reranking the hybrid top-20 puts the right chunk first for 21/42 questions
instead of 17**, and lifts MRR from 0.527 to 0.599 (+14% relative). Recall@5 goes
from 30/42 to 32/42. With 42 questions each question is 2.4 pp, so the Recall@5
gain is two questions and should be read as directional; the Recall@1 and MRR
gains are larger and matter most for Phase 7, which answers from the top chunks.

**Choosing the reranker.** The larger `bge-reranker-base` (278M params) was the
original plan. The 22M-param `ms-marco-MiniLM-L-6-v2` beat it on every metric at
~6.5× lower latency, so it is the default. `bge-reranker-base` gained 2 questions in
the top 5 but lost 3, pushing two gold chunks out of the top 10 entirely.

**Ceiling.** A reranker can only reorder what fusion hands it. The gold chunk is in
the hybrid top-20 for 36/42 questions (85.7%), so that is the maximum Recall@5 at
depth 20. Depth 50 raises the ceiling to 39/42 (92.9%) at ~2.5× the rerank cost.

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
                       · hybrid top-20 → MiniLM-L-6 → top-k
                                │
                                ├──> Phase 8: FastAPI /query (score gate + passages)
                                │
                                └──> Phase 7.5: NLI claim checker (separate CLI)
```

## API

Returns handbook passages, not generated answers, so it needs no API key.

```bash
pip install -e ".[api]"
uvicorn src.api:app --port 8000     # try it in the browser at http://127.0.0.1:8000/docs
```

`POST /query` takes a `question` and an optional `k` (1–20, default 5):

```json
{"question": "what is the SIRT on-call schedule", "k": 3}
```

```json
{
  "question": "what is the SIRT on-call schedule",
  "confident": true,
  "top_score": 6.86,
  "passages": [
    {
      "rank": 1,
      "score": 6.86,
      "chunk_id": "security/security-operations/secops-oncall.md::0000",
      "source_path": "security/security-operations/secops-oncall.md",
      "doc_title": "...",
      "section_heading": "...",
      "text": "..."
    }
  ]
}
```

`confident` means the top passage cleared the score gate. Below the gate, `passages` is
empty. Passing doesn't prove the passages answer the question (see caveats).
`GET /health` returns `{"status": "ok"}`.

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
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`
  (`BAAI/bge-reranker-base` selectable for comparison)
- **Verification:** `cross-encoder/nli-deberta-v3-base` via `sentence-transformers`
- **Generation (optional, never run live):** Claude API (`claude-opus-5-5`) with structured output
- **Serving:** FastAPI, Uvicorn

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
python -m src.rerank           "how do I take unpaid leave" --k 5   # hybrid + rerank

# 5. Run the eval (all four retrievers)
python -m eval.run_eval
python -m eval.run_eval --retriever rerank --rerank-model BAAI/bge-reranker-base

# 6. Check a statement against the corpus, and measure the checker
python -m src.verify "Unspent growth and development budgets roll over"
python -m eval.eval_verify --claims eval/claims_cli.json --k 5

# 7. Tests and the "I don't know" gate (no API calls or cost)
pip install -e ".[api,generation,dev]"
python -m eval.calibrate_gate
python -m pytest tests

# 8. Serve it (see API above)
uvicorn src.api:app --port 8000

# Optional: Claude answers (paid API; put ANTHROPIC_API_KEY in a .env file)
python -m src.generate "how do I take unpaid leave"
python -m src.generate "how do I take unpaid leave" --dry-run   # print prompt, no API call
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
| 6 | Cross-encoder reranking (MiniLM-L-6; MRR 0.527 → 0.599, Recall@1 40.5% → 50.0%) | ✅ |
| 7 | Optional generation with forced citations (Claude API); never run live, simulated API only | ⏸️ |
| 7.5 | NLI claim checker (held-out, top 5 chunks: kept 13/15 supported, caught 16/19 unsupported) | ✅ |
| 8 | FastAPI `/query` returning passages + score gate; no API key needed | ✅ |

## Known caveats

- **19.2 % of chunks exceed the BGE 512-token limit** and have their tails silently
  truncated by the dense encoder. Only 22 chunks are severely affected (>50 % lost —
  all giant tables that Phase 1 correctly refused to split). Phase 5 evals confirmed
  1 of the 10 hybrid misses is directly caused by this (1,537-token CorpSec
  Engineering chunk). The cross-encoder has the same 512-token input limit. Not yet fixed: with a single attributable miss, the 42-question
  set is too small to measure the lift, so the fix waits for a larger query set.
- **Some chunks contain raw HTML** (`<iframe>`, Font Awesome `<i>` tags) that isn't
  Hugo shortcode syntax and slipped past the cleaner. Roughly 5 % of chunks;
  BM25 tokenization treats it as noise. The Phase 5 miss analysis traced none of the
  10 hybrid misses to it, so cleanup is still open and low priority.
- **The gold questions were generated from each chunk's title and section heading**
  (`eval/generate_questions.py`), and the reranker sees title + heading + body.
  That may favour reranking more than real user queries would. A hand-written
  query set is the fix.
- **The retrieval gate only catches off-topic questions.** At a threshold of -2 it
  blocks 5/5 off-topic questions, 2/7 near-misses and 0/42 answerable questions
  (`eval/calibrate_gate.py`; small, single-use sample). "How long is GitLab's sabbatical
  leave?" scores 4.34 though the corpus never mentions sabbaticals, so retrieval alone
  returns passages for these questions.
- **The NLI checker makes mistakes, and its test set is small.** On 34 held-out claims
  (`eval/claims_cli.json`) checked against the top 5 reranked chunks, as the CLI does, it
  kept 13/15 supported claims and caught 16/19 unsupported ones (9/12 contradicted, 7/7 not
  stated). Against only the chunk each claim was written from, it kept 12/15 and caught
  19/19. I wrote the claims myself, so this measures the checker on hand-written statements,
  not the faithfulness of generated answers.
- **"Supported" from the CLI can still be wrong: 3 of the 16 held-out claims it passed were
  false.** Each matched a passage on a nearby topic: "Offboarding tasks can take up to 30
  days to complete" passed at 0.99 on a sentence about having 30 days to leave the UAE after
  a visa is cancelled. Raising the threshold from 0.5 to 0.9 cut these from 6 to 3, but these
  scores are too high for a threshold to remove. Always read the evidence it prints.
- **"No supporting passage found" doesn't mean false.** The checker also misses true claims:
  2 of the 15 held-out supported claims failed, likely because the fact is spread across a
  list or the passage lacks the title or heading that gives it context.
- **Very long chunks are slow to check.** Passages run at about 0.15 s each on CPU, so a
  typical chunk takes under 2 s and the largest (1,199 passages) takes about 3 minutes.
  This is why `/query` doesn't run the checker.
- **`confident: true` doesn't mean the handbook answers the question.** The sabbatical
  question above passes the gate at 4.34 and returns leave passages about other topics.
- **The API is for local use and demos.** It has no authentication or rate limiting, so
  don't expose it publicly. One running server used about 790 MB of memory and took 15–35 s
  to start. Requests took about 3 s each, slower than the 0.9 s reranker benchmark above;
  the laptop was short on free memory during this measurement, which may explain some of
  the gap. Three requests at once took 8.3 s, so they gain little from running together.

## Repo layout

```
src/
├── ingest.py              # Phase 1  — parse + chunk Markdown
├── index_bm25.py          # Phase 2  — BM25 index build + persist
├── retrieve_bm25.py       # Phase 2  — search_bm25() + CLI
├── index_dense.py         # Phase 3  — Chroma index build + persist
├── retrieve_dense.py      # Phase 3  — search_dense() + CLI
├── compare_retrievers.py  # Phase 3/4 — side-by-side retriever comparison
├── retrieve_hybrid.py     # Phase 4  — RRF fusion + CLI
├── rerank.py              # Phase 6  — cross-encoder rerank + CLI
├── generate.py            # Phase 7  — optional cited answers from Claude + CLI
├── verify.py              # Phase 7.5 — NLI claim checker + CLI
└── api.py                 # Phase 8  — FastAPI /query and /health

eval/
├── generate_questions.py  # Phase 5  — gold Q&A set generator
├── questions.json         # Phase 5  — 42 gold question → chunk_id pairs
├── run_eval.py            # Phase 5/6 — recall@1/5/10 + MRR + latency harness
├── unanswerable.json      # Phase 7  — 12 questions the corpus can't answer
├── calibrate_gate.py      # Phase 7  — measures the "I don't know" threshold
├── claims.json            # Phase 7.5 — 39 labelled claims, used to choose settings
├── claims_long.json       # Phase 7.5 — 18 labelled claims on long chunks
├── claims_cli.json        # Phase 7.5 — 34 labelled claims, held out for the final test
└── eval_verify.py         # Phase 7.5 — measures the claim checker

tests/
├── test_generate.py       # Phase 7  — generation tests against a simulated API
├── test_verify.py         # Phase 7.5 — checker logic with a fake model
└── test_api.py            # Phase 8  — API behaviour with retrieval faked

data/
├── raw/                   # source Markdown (gitignored, redownloadable)
└── processed/
    ├── chunks.json        # Phase 1 output (gitignored)
    └── bm25_index.pkl     # Phase 2 output (gitignored)

.chroma/                   # Phase 3 output (gitignored, ~39 MB)
PROGRESS.md                # detailed live tracker (open questions, per-phase notes)
```
