# Project Progress & Context

Living tracker for the Hybrid Search RAG project. Update the status boxes as phases complete.
This file also serves as context recovery: if a chat session loses history, this rebuilds it.

Last updated: 2026-11-02 (retrieval-only decision; Phase 7.5 done)

---

## Project in one line

Hybrid retrieval (BM25 + dense + RRF fusion + cross-encoder rerank) with a local NLI claim
checker over a subset of the public GitLab Handbook, with an eval harness that measures
retrieval recall. Retrieval-only: everything must run free, with no paid API calls.

## Locked decisions

| Decision | Choice | Reason |
|---|---|---|
| Corpus | GitLab Handbook subset (`security/`, `people-group/`, maybe `engineering/`) | Real internal-style docs, public, honest project title |
| Repo | https://github.com/aasheeeeesh/Hybrid-Search-RAG (public) | Resume-visible |
| Local project path | `C:\Users\AASHISH\projects\Hybrid-Search-RAG` | Outside OneDrive to avoid venv/git sync locks |
| Compute | Local, CPU only | Free, teaches more; no GPU available |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim, ~130 MB) | Fast on CPU, plenty for this corpus |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder | Beat `bge-reranker-base` on the gold set at ~6.5× lower latency (Phase 6) |
| NLI (verification) | `cross-encoder/nli-deberta-v3-base` | Local claim-vs-chunk entailment |
| Generation | None by default; `src/generate.py` kept as optional | No API credit, so the project is retrieval-only (2026-11-02) |
| No LangChain | Write pipeline from scratch (~400 lines) | Understanding + interview credibility |
| Framework | FastAPI endpoint; Streamlit UI last (optional) | Demo value after retrieval works |

## Resume line (true today)

> Hybrid RAG retrieval with BM25 + dense search, RRF fusion and cross-encoder
> reranking, measured by an eval harness on a 42-question gold set. Reranked hybrid
> reached Recall@5 of 76.2% (BM25 57.1%, dense 64.3%) and MRR@10 of 0.599 (BM25 0.458).

Don't claim working answer generation or answer faithfulness: `src/generate.py` has never
completed a live call. The NLI checker can be described as a claim checker measured on 18
held-out hand-written claims, a small sample.

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

- ✅ **Phase 3 — Dense retrieval**
  - ✅ `src/index_dense.py`: BAAI/bge-small-en-v1.5 (384-dim), Chroma
    PersistentClient at `.chroma/` (39 MB), hnsw:space=cosine, batch_size=32
  - ✅ Encoding: 2622 chunks in 199s on CPU (~13 chunks/sec)
  - ✅ `src/retrieve_dense.py`: `search_dense()` mirrors `search_bm25()` signature,
    module-level lazy singletons for model + collection (Phase 5 hot-loop friendly),
    distance → similarity conversion, prefix stripping on returned text
  - ✅ `src/compare_retrievers.py`: side-by-side BM25 vs Dense with overlap counter
  - ✅ Checkpoint (7 queries): **3 of 7 had 0/5 overlap** — empirical case for
    hybrid. Key wins:
    - "anti-harassment policy" → dense fixed the Phase 2 rank-3 bug (rank 1 now)
    - "who do I contact about workplace complaints" → dense: 5/5 relevant, BM25: 0/5
    - "unpaid leave" → 0/5 overlap, both find valid leave chunks (different)
  - ⚠️ Known: 19.2% of chunks exceed BGE's 512-token limit (silent tail
    truncation); of those, only 22 chunks lose >50% content. Deferred to
    Phase 5 evals — surgical fix if it surfaces as measurable retrieval loss.

- ✅ **Phase 4 — Hybrid fusion / RRF**
  - ✅ `src/retrieve_hybrid.py`: `search_hybrid(query, k, *, pool, bm25_hits,
    dense_hits)` implements RRF with k_rrf=60 (Cormack 2009 default)
  - ✅ Over-retrieval: pool=20 per retriever before fusing to k=5
  - ✅ CLI with --pool/--k validation (usage error exit code 2)
  - ✅ Same result-dict schema as BM25 and Dense — Phase 6 rerank and
    Phase 5 eval treat all three retrievers interchangeably
  - ✅ `src/compare_retrievers.py --hybrid`: 3-column view with composition
    report (BOTH / BM25-only / Dense-only / deeper-pool)
  - ✅ Checkpoint (7 queries): fusion behaves as designed on every query.
    - "anti-harassment policy" → hybrid rank 1 = anti-harassment.md::0000
      (the actual doc's Introduction). BM25 had it at rank 3, Dense at rank 2.
      RRF math: 1/63 + 1/62 = 0.0320 ✓
    - **6 of 35 hybrid slots (17%) came from the deeper pool** — chunks that
      were in *neither* retriever's top-5. Concrete evidence over-retrieval
      earns its cost.
    - High-overlap query (expense: 4/5) → hybrid returns 4 shared + 1 unique.
      Fusion never degrades on easy queries.

- ✅ **Phase 5 — Eval harness**
  - ✅ `eval/generate_questions.py`: samples diverse chunks (per-folder quota,
    one per source path, 150+ token filter), heuristic question generation
    from section headings + doc titles
  - ✅ `eval/questions.json`: 42 gold Q→chunk_id pairs, hand-curated questions
    (security: 22, people-group: 14, people-policies: 6)
  - ✅ `eval/run_eval.py`: recall@1/5/10 + MRR@10 for BM25 / Dense / Hybrid;
    --verbose per-query ranks, --misses flag; lazy singleton reuse (hot loop)
  - ✅ **Results (42 questions, k=10):**

    | Retriever | R@1 | R@5 | R@10 | MRR@10 |
    |---|:---:|:---:|:---:|:---:|
    | BM25 | 35.7% | 57.1% | 71.4% | 0.458 |
    | Dense (BGE) | 35.7% | 64.3% | 76.2% | 0.464 |
    | **Hybrid (RRF)** | **40.5%** | **71.4%** | **76.2%** | **0.527** |

  - ✅ Hybrid R@5 (+14.3 pp over BM25, +7.1 pp over Dense); MRR +15% relative
  - ✅ 10 hybrid misses analysed: 4 are explainable (1 over BGE 512-token limit,
    1 French text vs English query, 2 deep-nesting content mismatches)

- ✅ **Phase 6 — Cross-encoder reranking** (1 day)
  - ✅ `src/rerank.py`: hybrid top-20 → cross-encoder → top-k, CLI with `--depth`, `--model`
  - ✅ Ceiling: hybrid Recall@20 = 36/42 (85.7%), Recall@50 = 39/42 (92.9%)
  - ✅ MiniLM-L-6 (default): R@1 21/42, R@5 32/42, R@10 35/42, MRR 0.599, 0.9 s/query CPU
  - ✅ bge-reranker-base: R@1 21/42, R@5 29/42, R@10 34/42, MRR 0.578, 6.1 s/query —
    gained 2 in top 5, lost 3; dropped as default
  - ✅ Eval harness: `rerank` retriever, Hits@5 column, per-query latency (warmup excluded)
  - Note: running two rerank evals at once ran the machine out of memory; run one at a time

- ⏸️ **Phase 7 — Generation with citations** (optional; built, never run live)
  - ✅ `src/generate.py`: top reranked chunks → token-budgeted numbered context → Claude
    structured output (claims with document-id citations) → `[source §heading]` display
  - ✅ Claims with no valid citation are dropped and reported, never shown as answers
  - ✅ Oversized chunks are skipped and reported, not cut (largest chunk: 33,640 tokens)
  - ✅ `eval/calibrate_gate.py` + `eval/unanswerable.json` (12 questions, checked absent
    from the corpus): gate at -2 blocks 5/5 off-topic and 2/7 near-miss, 0/42 answerable
  - ✅ 17 tests pass against a simulated API; CLI gate and `--dry-run` run for real;
    the real API rejected an invalid key as expected
  - Not planned: a live run. No API credit is expected, so the API accepting the request,
    answer quality and near-miss handling stay unverified (last attempt stopped at
    "credit balance is too low").
  - Finding: the gate only handles off-topic questions. "How long is GitLab's sabbatical
    leave?" scores 4.34, above the lowest answerable question (3.02), though the corpus
    never mentions sabbaticals.

- ✅ **Phase 7.5 — NLI claim checker** (runs locally)
  - ✅ `src/verify.py`: splits a chunk into 3-sentence passages; a claim is supported if any
    passage entails it. `verify_claims` separates supported from unsupported claims and
    returns their share; CLI checks a statement against the top reranked chunks
  - ✅ `eval/eval_verify.py`: window chosen on 39 claims from short chunks, then tested on 18
    from long chunks at threshold 0.5: kept 7/9 supported, caught 8/9 unsupported. The whole
    chunk kept 5/9, caught 9/9
  - ✅ Threshold raised from 0.5 to 0.9 after the CLI said "Unspent growth and development
    budgets roll over" was supported (0.53, unrelated passage). Chosen on those 57 claims
    plus 4 roll-over phrasings, so passing those says little. Tested once on 34 new claims
    (`eval/claims_cli.json`, written before scoring) against the top 5 reranked chunks:
    - Threshold 0.5: kept 13/15 supported, caught 13/19 unsupported, precision 0.68
    - Threshold 0.9: kept 13/15 supported, caught 16/19 unsupported, precision 0.81
    - Threshold 0.9, only the chunk each claim came from: kept 12/15, caught 19/19
  - ✅ 13 logic tests with a fake model
  - Dropped: regenerating bad claims and a faithfulness score on real answers. Both need
    generated answers, which need the API. The share from `verify_claims` is untested on them.
  - Still wrong: 3 of the 16 held-out claims the CLI passed were false, each at 0.97 or above
    on a passage about a nearby topic, so a higher threshold won't remove them.
  - Tried and not used: adding title + heading to each passage scored one claim better on
    the 57 but checks every passage twice. A contradiction veto looks unusable: unrelated
    passages also score 0.98+ for contradiction. `eval/claims_cli.json` has now been seen, so
    testing another change needs new claims.

- ⬜ **Phase 8 — API + polish** (1–2 days)
  - FastAPI `POST /query` returning top passages with sources and the "I don't know" gate
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
- API key lives in `.env` (gitignored), loaded by `src/generate.py`.
- Windows: `import nltk` checks `D:\nltk_data`, so a failing USB drive on D: hangs anything
  importing BM25, with no output and no CPU use. Unplug the drive if this happens.
