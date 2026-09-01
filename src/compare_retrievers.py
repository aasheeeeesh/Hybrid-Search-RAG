# src/compare_retrievers.py
# Phase 3 — Side-by-side comparison of BM25 vs Dense retrieval.
#
# The single most instructive tool in this project. Run the same query
# through both retrievers and see, at a glance, where they agree,
# where they disagree, and why hybrid retrieval matters.
#
# Usage:
#   python -m src.compare_retrievers "your query"
#   python -m src.compare_retrievers "your query" --k 5

import argparse
import sys

from src.retrieve_bm25 import search_bm25
from src.retrieve_dense import search_dense

# Column widths tuned so two lists fit on a 160-char terminal.
COL_WIDTH = 78
TEXT_PREVIEW_CHARS = 180


def _format_hit(hit: dict, score_label: str) -> list[str]:
    """Return a list of lines (already width-limited) describing one hit."""
    preview = hit["text"][:TEXT_PREVIEW_CHARS].replace("\n", " ")
    if len(hit["text"]) > TEXT_PREVIEW_CHARS:
        preview += " ..."

    title = hit["doc_title"] or "(no title)"
    section = hit["section_heading"] or "(no heading)"

    lines = [
        f"[{hit['rank']}] {score_label}={hit['score']:.4f}  "
        f"id={hit['chunk_id']}",
        f"    T: {title}",
        f"    S: {section}",
        f"    {preview}",
    ]
    # Wrap each line to the column width (single hard cut, no word-wrap:
    # eyeballs cope fine and it keeps the two columns aligned).
    return [ln[:COL_WIDTH].ljust(COL_WIDTH) for ln in lines]


def _print_side_by_side(bm25_hits: list[dict], dense_hits: list[dict]) -> None:
    header_left = "BM25 (sparse / keyword)".center(COL_WIDTH)
    header_right = "Dense (semantic / bge-small)".center(COL_WIDTH)
    sep = " | "

    print(header_left + sep + header_right)
    print("=" * COL_WIDTH + sep + "=" * COL_WIDTH)

    for bhit, dhit in zip(bm25_hits, dense_hits):
        bcol = _format_hit(bhit, "score")
        dcol = _format_hit(dhit, "sim")
        for bl, dl in zip(bcol, dcol):
            print(bl + sep + dl)
        print("-" * COL_WIDTH + sep + "-" * COL_WIDTH)


def _overlap_report(bm25_hits: list[dict], dense_hits: list[dict]) -> None:
    bm25_ids = {h["chunk_id"] for h in bm25_hits}
    dense_ids = {h["chunk_id"] for h in dense_hits}
    shared = bm25_ids & dense_ids
    only_bm25 = bm25_ids - dense_ids
    only_dense = dense_ids - bm25_ids

    print()
    print(f"Overlap in top-{len(bm25_hits)}: {len(shared)} chunk(s) appear in both lists.")
    if shared:
        print("  Shared:")
        for cid in sorted(shared):
            print(f"    - {cid}")
    if only_bm25:
        print(f"  BM25 only ({len(only_bm25)}):")
        for cid in sorted(only_bm25):
            print(f"    - {cid}")
    if only_dense:
        print(f"  Dense only ({len(only_dense)}):")
        for cid in sorted(only_dense):
            print(f"    - {cid}")


def compare(query: str, k: int = 5) -> tuple[list[dict], list[dict]]:
    """Run both retrievers on `query` and return their (bm25, dense) result lists."""
    bm25_hits = search_bm25(query, k=k)
    dense_hits = search_dense(query, k=k)
    return bm25_hits, dense_hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of BM25 vs dense retrieval."
    )
    parser.add_argument("query", help="Search query string.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of results per retriever (default: 5).")
    args = parser.parse_args()

    print(f'\nQuery: "{args.query}"\n')
    bm25_hits, dense_hits = compare(args.query, k=args.k)

    if not bm25_hits and not dense_hits:
        print("Both retrievers returned nothing.", file=sys.stderr)
        sys.exit(1)

    # Pad the shorter list so the side-by-side loop doesn't stop early.
    while len(bm25_hits) < args.k:
        bm25_hits.append({
            "rank": len(bm25_hits) + 1, "score": 0.0, "chunk_id": "(none)",
            "text": "(no result)", "doc_title": "", "section_heading": ""
        })
    while len(dense_hits) < args.k:
        dense_hits.append({
            "rank": len(dense_hits) + 1, "score": 0.0, "chunk_id": "(none)",
            "text": "(no result)", "doc_title": "", "section_heading": ""
        })

    _print_side_by_side(bm25_hits, dense_hits)
    _overlap_report(bm25_hits, dense_hits)
