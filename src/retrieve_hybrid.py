# src/retrieve_hybrid.py
# Phase 4 — Hybrid retrieval via Reciprocal Rank Fusion.
#
# Combines BM25 (sparse/keyword) and dense (semantic) retrieval by summing
# RRF contributions across both ranked lists. Retriever-agnostic: only the
# per-retriever *ranks* matter, not the incomparable raw scores.
#
# Provides:
#   search_hybrid(query, k, *, pool, bm25_hits, dense_hits) -> list[dict]
#
# RRF formula (Cormack, Clarke, Buettcher 2009):
#   score(chunk) = Σ  1 / (k_rrf + rank_i(chunk))
#                 i∈retrievers
# where k_rrf=60 is the paper default and rank_i is 1-indexed.

import argparse
import sys

from src.retrieve_bm25 import search_bm25
from src.retrieve_dense import search_dense

# The RRF dampening constant. 60 is the value from the original paper —
# it's not really tunable, its role is to prevent rank 1 from dominating
# so completely that consensus at ranks 3-5 can never overtake it.
RRF_K = 60

# How many candidates to retrieve from each retriever before fusion.
# We deliberately over-retrieve: a chunk at BM25 rank 6 + dense rank 3
# can beat some top-5 chunks after fusion, but only if we actually fetched
# it. 20 is a safe default; the cost is small since neither retriever
# scales badly at k=20 vs k=5.
DEFAULT_POOL = 20


def search_hybrid(
    query: str,
    k: int = 5,
    *,
    pool: int = DEFAULT_POOL,
    bm25_hits: list[dict] | None = None,
    dense_hits: list[dict] | None = None,
) -> list[dict]:
    """Return the top-k RRF-fused results for *query*.

    Parameters
    ----------
    query       : Natural-language query string.
    k           : Number of results to return after fusion.
    pool        : Depth to retrieve from each retriever before fusing.
                  Must be >= k (over-retrieval improves recall).
    bm25_hits   : Optional pre-computed BM25 results (Phase 5 hot loop
                  passes these to avoid duplicate work).
    dense_hits  : Optional pre-computed dense results (same rationale).

    Returns
    -------
    List of dicts sorted by descending RRF score. Same schema as
    search_bm25 / search_dense so downstream consumers (Phase 6 rerank,
    Phase 5 eval) treat all three retrievers interchangeably. The `score`
    field is the RRF sum, not a similarity — no natural scale, only
    relative ranking is meaningful.
    """
    if bm25_hits is None:
        bm25_hits = search_bm25(query, k=pool)
    if dense_hits is None:
        dense_hits = search_dense(query, k=pool)

    # Accumulate per-chunk RRF score and keep the first-seen metadata
    # (whichever retriever hit it first — the fields we surface are
    # identical across retrievers, so this is a no-op for display).
    scores: dict[str, float] = {}
    hit_meta: dict[str, dict] = {}

    for hit in bm25_hits:
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + hit["rank"])
        hit_meta.setdefault(cid, hit)

    for hit in dense_hits:
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + hit["rank"])
        hit_meta.setdefault(cid, hit)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for rank, (cid, score) in enumerate(ranked, start=1):
        meta = hit_meta[cid]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": cid,
                "token_count": meta["token_count"],
                "doc_title": meta["doc_title"],
                "section_heading": meta["section_heading"],
                "text": meta["text"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# CLI pretty-printer — matches the shape of retrieve_bm25 / retrieve_dense
# ---------------------------------------------------------------------------

TEXT_PREVIEW_CHARS = 200


def _print_results(results: list[dict], query: str) -> None:
    print(f'\nHybrid (RRF) results for: "{query}"')
    print("=" * 70)
    for r in results:
        preview = r["text"][:TEXT_PREVIEW_CHARS]
        if len(r["text"]) > TEXT_PREVIEW_CHARS:
            preview += " ..."
        print(
            f"[{r['rank']}] rrf={r['score']:.4f}  "
            f"tokens={r['token_count']}  id={r['chunk_id']}"
        )
        print(f"     Title  : {r['doc_title']}")
        print(f"     Section: {r['section_heading']}")
        print(f"     Text   : {preview}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query BM25 + dense retrievers and print RRF-fused top-k."
    )
    parser.add_argument("query", help="Search query string.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of results to return after fusion (default: 5).")
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL,
                        help=f"Depth to retrieve from each retriever before fusion "
                             f"(default: {DEFAULT_POOL}).")
    args = parser.parse_args()

    if args.pool < args.k:
        print(f"ERROR: --pool ({args.pool}) must be >= --k ({args.k}).",
              file=sys.stderr)
        sys.exit(2)

    results = search_hybrid(args.query, k=args.k, pool=args.pool)

    if not results:
        print(f'No results returned for "{args.query}".', file=sys.stderr)
        sys.exit(1)

    _print_results(results, args.query)
