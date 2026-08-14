# Project Progress & Context

Living tracker for the Hybrid Search RAG project. Update the status boxes as phases complete.
This file also serves as context recovery: if a chat session loses history, this rebuilds it.

Last updated: 2026-07-23 (end of Phase 0)

---

## Project in one line

Hybrid retrieval RAG (BM25 + dense + RRF fusion + cross-encoder rerank + NLI citation
verification) over a subset of the public GitLab Handbook, with an eval harness that
measures retrieval recall and answer faithfulness.

## Locked decisions

| Decision | Choice | Reason |
|---|---|---|
| Corpus | GitLab Handbook subset (`security/`, `people-group/`, maybe `engineering/`) | Real internal-style docs, public, honest project title |
| Repo | https://github.com/aasheeeeesh/Hybrid-Search-RAG (public) | Resume-visible |
| Local project path | `C:\Users\AASHISH\projects\Hybrid-Search-RAG` | Outside OneDrive to avoid venv/git sync locks |
| Compute | Local, CPU only | Free, teaches more; no GPU available |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim, ~130 MB) | Fast on CPU, plenty for this corpus |
| Reranker | `bge-reranker-base` cross-encoder | Standard strong CPU-friendly reranker |
| NLI (verification) | `cross-encoder/nli-deberta-v3-base` | Local claim-vs-chunk entailment |
| Generation | Claude API | Best answers; API key added later |
| No LangChain | Write pipeline from scratch (~400 lines) | Understanding + interview credibility |
| Framework | FastAPI endpoint; Streamlit UI last (optional) | Demo value after retrieval works |

## Working method

- **User writes the RAG code** (ingest/retrieve/rerank/etc.) — that's the learning.
- Claude guides, teaches each tool as it's introduced, reviews, unblocks.
- Boilerplate/scaffolding (repo skeleton, config) Claude may do.

## Resume line (target)

> Hybrid RAG system with BM25 + dense retrieval, RRF fusion, cross-encoder reranking,
> and NLI-based citation verification; eval harness measuring retrieval recall@5 and
> answer faithfulness. Hybrid retrieval improved recall@5 from X% to Y% over vector-only.

Fill in X and Y after Phase 5/6.

---

## Phase tracker

Legend: ✅ done · 🔄 in progress · ⬜ not started

- ✅ **Phase 0 — Repo + skeleton**
  - ✅ Local folder created outside OneDrive
  - ✅ Git init, `.gitignore`, `README.md`, `pyproject.toml`, `src/`, `eval/`
  - ✅ First commit pushed to GitHub (`main`)
  - ✅ venv created (Python 3.11.9)
  - ✅ `pip install -e .` — all imports verified (torch 2.13 CPU, chromadb 1.5.9, st 5.6.0)
  - ✅ Corpus: `security/` + `people-group/` + `people-policies/` → `data/raw/`
  - ✅ Count: **556 .md files, 5.1 MB** (engineering/ excluded — 1209 files, too slow for CPU embedding)

- ✅ **Phase 1 — Ingestion & Chunking**
  - ✅ Parse MD → plain text; keep front matter `title` as metadata (python-frontmatter)
  - ✅ Strip Hugo shortcodes: `{{% ... %}}` and `{{< ... >}}` blocks
  - ✅ Heading-aware chunking on `##`/`###`, ~300–500 tokens (tiktoken-measured)
  - ✅ Never split a markdown table mid-chunk; drop link targets, keep link text
  - ✅ Attach metadata: source path, section heading, chunk id
  - ✅ Output: one JSON of chunks in `data/processed/`
  - ✅ Checkpoint: any chunk reads as a coherent standalone passage
  - ✅ Note: many topics are folders with `_index.md` + sub-pages (Hugo); glob `**/*.md`

- ✅ **Phase 2 — BM25 retrieval**
  - ✅ `src/index_bm25.py`: conservative tokenizer (lowercase, split whitespace +
    `.,!?;:()[]{}<>"'/\`+=*&^%$~`, preserve `-` and `_`), Snowball stemmer
  - ✅ Searchable text = `doc_title + section_heading + text` (headings weighted
    naturally by appearing in indexed text)
  - ✅ Pickled index at `data/processed/bm25_index.pkl` (7.3 MB, 2622 chunks, ~14s build)
  - ✅ `src/retrieve_bm25.py`: `search_bm25(query, k, *, bm25, chunks)` importable
    for Phase 5 eval hot-loop; CLI wrapper prints top-k with metadata
  - ✅ Checkpoint queries (5 tested):
    - `"expense reimbursement"`, `"onboarding laptop linux"`, `"gitlab-com/gl-security"`,
      `"how do I take unpaid leave"` → all top-5 highly relevant
    - `"anti-harassment policy"` → correct doc at rank 3 (loss to link-only chunks
      that mention both terms in-body). Expected BM25 weakness; hybrid + rerank
      will address in later phases.
    - Original "sabbatical" test query dropped: term not present in this corpus
      subset (would have been in `engineering/` which we excluded).

- ⬜ **Phase 3 — Dense retrieval** (1–2 days)
  - Embed chunks with bge-small, store in Chroma
  - CLI shows BM25 vs dense side by side
  - Checkpoint: see which queries each retriever wins

- ⬜ **Phase 4 — Hybrid fusion / RRF** (½ day)
  - Reciprocal Rank Fusion: score = Σ 1/(60 + rank)
  - Checkpoint: hybrid ≥ better single retriever on every test query

- ⬜ **Phase 5 — Eval harness** (1–2 days, reused after)
  - 30–50 question → gold chunk id pairs in `eval/questions.json`
  - Compute recall@5 for BM25 / dense / hybrid
  - Record results table → README

- ⬜ **Phase 6 — Cross-encoder reranking** (1 day)
  - Retrieve top-20 hybrid → rerank → top-5
  - Rerun evals, measure lift

- ⬜ **Phase 7 — Generation with citations** (1 day)
  - Claude answers from chunks, forced `[source §heading]` citations
  - "I don't know" path on low retrieval confidence

- ⬜ **Phase 7.5 — Citation verification** (1–2 days)
  - Tier 1: structural (cited ids exist in context, claims have citations)
  - Tier 2: NLI entailment (claim vs cited chunk)
  - Tier 3: flag/strip/regenerate unsupported claims; return faithfulness field
  - Add faithfulness metric to eval harness

- ⬜ **Phase 8 — API + polish** (1–2 days)
  - FastAPI `POST /query`
  - README with eval table + diagram
  - Optional Streamlit UI

**Estimated total:** ~2–3 weeks part-time.

---

## Open questions (answer to proceed)

1. Does the user know what BM25 computes (TF-IDF family, term-frequency saturation,
   doc-length normalization), or black box? → decides if Phase 2 opens with theory.
2. Confirm `pip install -e .` succeeded and corpus cloned (paste `.md` count).

## Notes / gotchas

- OneDrive path is forbidden for the working project (sync locks on venv/git).
- Git commit email: `aashish.arya06@gmail.com` — must be verified on GitHub for
  contribution graph to show green squares.
- `data/` is gitignored; corpus is redownloaded via script (script is part of the repo).
