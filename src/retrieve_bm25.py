# src/retrieve_bm25.py
# Phase 2 — BM25 Retrieval
#
# Provides:
#   search_bm25(query, k, bm25, chunks) → list[dict]   ← importable function
#
# CLI usage:
#   python -m src.retrieve_bm25 "your query here"
#   python -m src.retrieve_bm25 "your query here" --k 10

import argparse
import sys

from rank_bm25 import BM25Okapi

from src.index_bm25 import INDEX_PATH, load_index, tokenize

# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------


def search_bm25(
    query: str,
    k: int = 5,
    *,
    bm25: BM25Okapi | None = None,
    chunks: list[dict] | None = None,
    index_path: str = INDEX_PATH,
) -> list[dict]:
    """Return the top-k BM25 results for *query*.

    Parameters
    ----------
    query       : Natural-language query string.
    k           : Number of results to return.
    bm25        : Pre-loaded BM25Okapi object (optional — avoids re-loading disk
                  index when calling programmatically many times in a loop).
    chunks      : Corresponding chunk metadata list (must be passed together
                  with *bm25* if either is provided).
    index_path  : Path to the pickle file (used only when bm25/chunks are None).

    Returns
    -------
    List of dicts, one per result, sorted by descending BM25 score:
        rank            int     (1-indexed)
        score           float
        chunk_id        str
        token_count     int
        doc_title       str
        section_heading str
        text            str
    """
    if bm25 is None or chunks is None:
        bm25, chunks = load_index(index_path)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # argsort descending — take top-k
    import numpy as np  # rank_bm25 already depends on numpy; safe to import here

    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[idx]
        results.append(
            {
                "rank": rank,
                "score": float(scores[idx]),
                "chunk_id": chunk["chunk_id"],
                "token_count": chunk["token_count"],
                "doc_title": chunk.get("doc_title", ""),
                "section_heading": chunk.get("section_heading", ""),
                "text": chunk["text"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# CLI pretty-printer
# ---------------------------------------------------------------------------

TEXT_PREVIEW_CHARS = 200


def _print_results(results: list[dict], query: str) -> None:
    print(f'\nBM25 results for: "{query}"')
    print("=" * 70)
    for r in results:
        preview = r["text"][:TEXT_PREVIEW_CHARS]
        if len(r["text"]) > TEXT_PREVIEW_CHARS:
            preview += " …"
        print(
            f"[{r['rank']}] score={r['score']:.4f}  "
            f"tokens={r['token_count']}  id={r['chunk_id']}"
        )
        print(f"     Title  : {r['doc_title']}")
        print(f"     Section: {r['section_heading']}")
        print(f"     Text   : {preview}")
        print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the BM25 index and print the top-k results."
    )
    parser.add_argument("query", help="Search query string.")
    parser.add_argument(
        "--k", type=int, default=5, help="Number of results to return (default: 5)."
    )
    parser.add_argument(
        "--index",
        default=INDEX_PATH,
        help="Path to the BM25 pickle file (default: %(default)s).",
    )
    args = parser.parse_args()

    results = search_bm25(args.query, k=args.k, index_path=args.index)

    if not results or results[0]["score"] == 0.0:
        print(f'No matching results found for "{args.query}".', file=sys.stderr)
        sys.exit(1)

    _print_results(results, args.query)
