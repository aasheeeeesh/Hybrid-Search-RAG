# src/compare_retrievers.py
# Phase 3 (2-column) + Phase 4 (3-column) — side-by-side retriever comparison.
#
# Run the same query through BM25, Dense, and optionally Hybrid (RRF)
# and see, per row, where each retriever agrees and where each diverges.
#
# Usage:
#   python -m src.compare_retrievers "your query"                # 2-column
#   python -m src.compare_retrievers "your query" --hybrid       # 3-column
#   python -m src.compare_retrievers "your query" --k 5 --pool 20

import argparse
import sys

from src.retrieve_bm25 import search_bm25
from src.retrieve_dense import search_dense
from src.retrieve_hybrid import DEFAULT_POOL, search_hybrid

# Column widths. 60 fits three columns on a 200-char terminal;
# 78 fits two columns on a 160-char terminal.
COL_WIDTH_2 = 78
COL_WIDTH_3 = 60
TEXT_PREVIEW_CHARS_2 = 180
TEXT_PREVIEW_CHARS_3 = 130


def _format_hit(hit: dict, score_label: str, col_width: int, preview_chars: int) -> list[str]:
    """Return a list of lines (already width-limited) describing one hit."""
    preview = hit["text"][:preview_chars].replace("\n", " ")
    if len(hit["text"]) > preview_chars:
        preview += " ..."

    title = hit["doc_title"] or "(no title)"
    section = hit["section_heading"] or "(no heading)"

    lines = [
        f"[{hit['rank']}] {score_label}={hit['score']:.4f}  id={hit['chunk_id']}",
        f"    T: {title}",
        f"    S: {section}",
        f"    {preview}",
    ]
    return [ln[:col_width].ljust(col_width) for ln in lines]


def _pad_hits(hits: list[dict], k: int) -> list[dict]:
    """Ensure the list has exactly k entries; pad with (no result) placeholders."""
    while len(hits) < k:
        hits.append({
            "rank": len(hits) + 1, "score": 0.0, "chunk_id": "(none)",
            "text": "(no result)", "doc_title": "", "section_heading": "",
            "token_count": 0,
        })
    return hits


def _print_two_way(bm25_hits: list[dict], dense_hits: list[dict]) -> None:
    header_left = "BM25 (sparse / keyword)".center(COL_WIDTH_2)
    header_right = "Dense (semantic / bge-small)".center(COL_WIDTH_2)
    sep = " | "

    print(header_left + sep + header_right)
    print("=" * COL_WIDTH_2 + sep + "=" * COL_WIDTH_2)

    for bhit, dhit in zip(bm25_hits, dense_hits):
        bcol = _format_hit(bhit, "score", COL_WIDTH_2, TEXT_PREVIEW_CHARS_2)
        dcol = _format_hit(dhit, "sim", COL_WIDTH_2, TEXT_PREVIEW_CHARS_2)
        for bl, dl in zip(bcol, dcol):
            print(bl + sep + dl)
        print("-" * COL_WIDTH_2 + sep + "-" * COL_WIDTH_2)


def _print_three_way(bm25_hits, dense_hits, hybrid_hits) -> None:
    hL = "BM25 (sparse)".center(COL_WIDTH_3)
    hM = "Dense (semantic)".center(COL_WIDTH_3)
    hR = "Hybrid (RRF fusion)".center(COL_WIDTH_3)
    sep = " | "

    print(hL + sep + hM + sep + hR)
    print("=" * COL_WIDTH_3 + sep + "=" * COL_WIDTH_3 + sep + "=" * COL_WIDTH_3)

    for bhit, dhit, hhit in zip(bm25_hits, dense_hits, hybrid_hits):
        bcol = _format_hit(bhit, "score", COL_WIDTH_3, TEXT_PREVIEW_CHARS_3)
        dcol = _format_hit(dhit, "sim", COL_WIDTH_3, TEXT_PREVIEW_CHARS_3)
        hcol = _format_hit(hhit, "rrf", COL_WIDTH_3, TEXT_PREVIEW_CHARS_3)
        for bl, dl, hl in zip(bcol, dcol, hcol):
            print(bl + sep + dl + sep + hl)
        print("-" * COL_WIDTH_3 + sep + "-" * COL_WIDTH_3 + sep + "-" * COL_WIDTH_3)


def _overlap_two_way(bm25_hits, dense_hits) -> None:
    b, d = _ids(bm25_hits), _ids(dense_hits)
    shared = b & d
    print()
    print(f"Overlap in top-{len(bm25_hits)}: {len(shared)} chunk(s) in both lists.")
    _dump_group("Shared", sorted(shared))
    _dump_group("BM25 only", sorted(b - d))
    _dump_group("Dense only", sorted(d - b))


def _overlap_three_way(bm25_hits, dense_hits, hybrid_hits) -> None:
    """Where did Hybrid's top-k come from? Report the source of each hybrid pick."""
    b, d, h = _ids(bm25_hits), _ids(dense_hits), _ids(hybrid_hits)

    from_both = h & b & d
    from_bm25_only = h & (b - d)
    from_dense_only = h & (d - b)
    from_neither = h - b - d  # can happen when pool > k (chunk was in the pool but neither top-k)

    print()
    print(f"Hybrid top-{len(hybrid_hits)} composition:")
    print(f"  from BOTH BM25 & Dense top-k : {len(from_both)}")
    print(f"  from BM25 top-k only         : {len(from_bm25_only)}")
    print(f"  from Dense top-k only        : {len(from_dense_only)}")
    print(f"  from deeper pool (not in top-k of either) : {len(from_neither)}")

    print()
    print(f"BM25 <> Dense overlap in top-{len(bm25_hits)}: {len(b & d)} chunk(s).")
    _dump_group("Shared (BM25 & Dense)", sorted(b & d))
    _dump_group("BM25 only", sorted(b - d))
    _dump_group("Dense only", sorted(d - b))


def _ids(hits: list[dict]) -> set:
    return {h["chunk_id"] for h in hits if h["chunk_id"] != "(none)"}


def _dump_group(label: str, ids: list[str]) -> None:
    if not ids:
        return
    print(f"  {label} ({len(ids)}):")
    for cid in ids:
        print(f"    - {cid}")


def compare(query: str, k: int = 5, pool: int = DEFAULT_POOL, include_hybrid: bool = False):
    """Run selected retrievers and return their result lists.

    When include_hybrid is True, we retrieve top-`pool` from BM25 and Dense
    (needed for RRF over-retrieval), then slice the top-k for display in
    the BM25 / Dense columns while passing the full pool to search_hybrid.
    This avoids duplicate retriever calls.
    """
    if include_hybrid:
        bm25_full = search_bm25(query, k=pool)
        dense_full = search_dense(query, k=pool)
        hybrid_hits = search_hybrid(
            query, k=k, pool=pool,
            bm25_hits=bm25_full, dense_hits=dense_full,
        )
        return bm25_full[:k], dense_full[:k], hybrid_hits

    return search_bm25(query, k=k), search_dense(query, k=k), None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of BM25 vs Dense (and optionally Hybrid) retrieval."
    )
    parser.add_argument("query", help="Search query string.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of results per retriever (default: 5).")
    parser.add_argument("--hybrid", action="store_true",
                        help="Also show RRF-fused hybrid column.")
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL,
                        help=f"With --hybrid, depth to retrieve from each retriever "
                             f"before fusion (default: {DEFAULT_POOL}).")
    args = parser.parse_args()

    if args.hybrid and args.pool < args.k:
        print(f"ERROR: --pool ({args.pool}) must be >= --k ({args.k}).",
              file=sys.stderr)
        sys.exit(2)

    print(f'\nQuery: "{args.query}"\n')
    bm25_hits, dense_hits, hybrid_hits = compare(
        args.query, k=args.k, pool=args.pool, include_hybrid=args.hybrid,
    )

    if not bm25_hits and not dense_hits:
        print("Both retrievers returned nothing.", file=sys.stderr)
        sys.exit(1)

    _pad_hits(bm25_hits, args.k)
    _pad_hits(dense_hits, args.k)

    if args.hybrid:
        _pad_hits(hybrid_hits, args.k)
        _print_three_way(bm25_hits, dense_hits, hybrid_hits)
        _overlap_three_way(bm25_hits, dense_hits, hybrid_hits)
    else:
        _print_two_way(bm25_hits, dense_hits)
        _overlap_two_way(bm25_hits, dense_hits)
