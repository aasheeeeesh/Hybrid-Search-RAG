# src/rerank.py
# Phase 6 — Cross-encoder reranking.
#
# Takes the hybrid (RRF) candidate list and re-scores each (query, chunk)
# pair jointly with a cross-encoder, then keeps the top-k. Unlike the
# bi-encoder in Phase 3, the cross-encoder reads query and chunk together,
# so it can judge relevance directly — but it costs one forward pass per
# candidate, which is why it only sees the fused top-N, never the corpus.
#
# Provides:
#   rerank(query, candidates, k, *, model) -> list[dict]
#   search_reranked(query, k, *, depth)    -> list[dict]
#
# CLI usage:
#   python -m src.rerank "your query"
#   python -m src.rerank "your query" --k 10 --depth 50

import argparse
import sys

from sentence_transformers import CrossEncoder

from src.retrieve_hybrid import search_hybrid

# Default: 22M params. On the 42-question gold set (depth 20, CPU) it beat
# bge-reranker-base on every metric — Recall@5 32/42 vs 29/42, MRR 0.599 vs
# 0.578 — at ~0.9 s/query instead of ~6.1 s/query.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Larger alternative (278M params), kept for comparison runs via --model.
LARGE_MODEL_NAME = "BAAI/bge-reranker-base"

# Cross-encoder input limit (query + chunk together). Longer pairs are
# truncated from the chunk side — same 512-token ceiling as the BGE encoder.
MAX_LENGTH = 512

# How many hybrid candidates to rerank. A reranker can only reorder what it
# is given, so this sets the recall ceiling: hybrid Recall@20 = 85.7%,
# Recall@50 = 92.9% on the 42-question gold set.
DEFAULT_DEPTH = 20

# ---------------------------------------------------------------------------
# Lazy singletons, one per model name — same rationale as
# retrieve_dense.get_model().
# ---------------------------------------------------------------------------

_MODELS: dict[str, CrossEncoder] = {}


def get_model(name: str = MODEL_NAME) -> CrossEncoder:
    if name not in _MODELS:
        _MODELS[name] = CrossEncoder(name, max_length=MAX_LENGTH)
    return _MODELS[name]


def _passage(hit: dict) -> str:
    # Title + heading give the reranker the same context BM25 indexes.
    return f"{hit['doc_title']} — {hit['section_heading']}\n{hit['text']}"


def rerank(
    query: str,
    candidates: list[dict],
    k: int = 5,
    *,
    model: CrossEncoder | str | None = None,
) -> list[dict]:
    """Re-score *candidates* with the cross-encoder and return the top-k.

    Returns the same schema as search_hybrid, with `score` replaced by the
    cross-encoder logit (higher = more relevant; no fixed scale) and
    `fused_rank` holding the candidate's rank before reranking. *model* is
    a loaded CrossEncoder or a model name (default: MODEL_NAME).
    """
    if not candidates:
        return []
    if not isinstance(model, CrossEncoder):
        model = get_model(model or MODEL_NAME)

    scores = model.predict([(query, _passage(c)) for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for rank, i in enumerate(order, start=1):
        c = candidates[i]
        results.append(
            {
                **c,
                "rank": rank,
                "score": float(scores[i]),
                "fused_rank": c["rank"],
            }
        )
    return results


def search_reranked(
    query: str,
    k: int = 5,
    *,
    depth: int = DEFAULT_DEPTH,
    model: str = MODEL_NAME,
) -> list[dict]:
    """Hybrid top-*depth* → cross-encoder → top-k."""
    candidates = search_hybrid(query, k=depth, pool=depth)
    return rerank(query, candidates, k=k, model=model)


# ---------------------------------------------------------------------------
# CLI pretty-printer — matches the shape of retrieve_hybrid
# ---------------------------------------------------------------------------

TEXT_PREVIEW_CHARS = 200


def _print_results(results: list[dict], query: str) -> None:
    print(f'\nReranked results for: "{query}"')
    print("=" * 70)
    for r in results:
        preview = r["text"][:TEXT_PREVIEW_CHARS]
        if len(r["text"]) > TEXT_PREVIEW_CHARS:
            preview += " ..."
        print(
            f"[{r['rank']}] ce={r['score']:.3f}  (was #{r['fused_rank']})  "
            f"tokens={r['token_count']}  id={r['chunk_id']}"
        )
        print(f"     Title  : {r['doc_title']}")
        print(f"     Section: {r['section_heading']}")
        print(f"     Text   : {preview}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval followed by cross-encoder reranking."
    )
    parser.add_argument("query", help="Search query string.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of results to return after reranking (default: 5).")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help=f"Hybrid candidates to rerank (default: {DEFAULT_DEPTH}).")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Cross-encoder model name (default: {MODEL_NAME}).")
    args = parser.parse_args()

    if args.depth < args.k:
        print(f"ERROR: --depth ({args.depth}) must be >= --k ({args.k}).",
              file=sys.stderr)
        sys.exit(2)

    results = search_reranked(args.query, k=args.k, depth=args.depth,
                               model=args.model)

    if not results:
        print(f'No results returned for "{args.query}".', file=sys.stderr)
        sys.exit(1)

    _print_results(results, args.query)
