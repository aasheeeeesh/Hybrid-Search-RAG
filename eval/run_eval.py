# eval/run_eval.py
# Phase 5 — Retrieval evaluation harness.
#
# Loads eval/questions.json (42 gold Q->chunk_id pairs), runs each question
# through BM25 / Dense / Hybrid / Hybrid+rerank retrievers, and reports:
#
#   Recall@1  — gold chunk is rank 1
#   Recall@5  — gold chunk is in top 5  (primary resume metric)
#   Recall@10 — gold chunk is in top 10
#   MRR@10    — mean reciprocal rank across all queries
#
# Usage:
#   python -m eval.run_eval              # all three retrievers, k=10
#   python -m eval.run_eval --retriever bm25
#   python -m eval.run_eval --retriever dense
#   python -m eval.run_eval --retriever hybrid
#   python -m eval.run_eval --retriever rerank --rerank-depth 50
#   python -m eval.run_eval --retriever rerank --rerank-model BAAI/bge-reranker-base
#   python -m eval.run_eval --verbose    # print per-query detail

import argparse
import json
import sys
import time
from pathlib import Path

from src.retrieve_bm25 import search_bm25
from src.retrieve_dense import search_dense
from src.rerank import DEFAULT_DEPTH, MODEL_NAME, rerank
from src.retrieve_hybrid import DEFAULT_POOL, search_hybrid

QUESTIONS_PATH = Path("eval/questions.json")
EVAL_K = 10          # retrieve this many candidates per query
HYBRID_POOL = max(DEFAULT_POOL, EVAL_K)   # always >= EVAL_K


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def reciprocal_rank(results: list[dict], gold_id: str) -> float:
    for r in results:
        if r["chunk_id"] == gold_id:
            return 1.0 / r["rank"]
    return 0.0


def hit_at_k(results: list[dict], gold_id: str, k: int) -> bool:
    return any(r["chunk_id"] == gold_id for r in results if r["rank"] <= k)


# ---------------------------------------------------------------------------
# Per-retriever runner
# ---------------------------------------------------------------------------

def _run_one(
    name: str,
    questions: list[dict],
    verbose: bool,
    rerank_depth: int = DEFAULT_DEPTH,
    rerank_model: str = MODEL_NAME,
) -> dict:
    """Run all questions through *name* retriever; return metrics dict."""
    hits1 = hits5 = hits10 = rr_sum = 0
    n = len(questions)
    misses = []

    # Warm up lazy singletons before timing the loop
    _ = search_bm25("warmup", k=1) if name in ("bm25", "hybrid", "rerank") else None
    _ = search_dense("warmup", k=1) if name in ("dense", "hybrid", "rerank") else None
    if name == "rerank":
        _ = rerank("warmup", [{"rank": 1, "doc_title": "", "section_heading": "",
                               "text": "warmup"}], k=1, model=rerank_model)
    # Warmup above is excluded from timing
    t0 = time.perf_counter()

    for q in questions:
        query = q["question"]
        gold  = q["gold_chunk_id"]

        if name == "bm25":
            results = search_bm25(query, k=EVAL_K)
        elif name == "dense":
            results = search_dense(query, k=EVAL_K)
        elif name == "hybrid":
            b = search_bm25(query, k=HYBRID_POOL)
            d = search_dense(query, k=HYBRID_POOL)
            results = search_hybrid(query, k=EVAL_K, pool=HYBRID_POOL,
                                    bm25_hits=b, dense_hits=d)
        else:  # rerank: hybrid top-rerank_depth -> cross-encoder -> top-EVAL_K
            b = search_bm25(query, k=rerank_depth)
            d = search_dense(query, k=rerank_depth)
            cands = search_hybrid(query, k=rerank_depth, pool=rerank_depth,
                                  bm25_hits=b, dense_hits=d)
            results = rerank(query, cands, k=EVAL_K, model=rerank_model)

        rr   = reciprocal_rank(results, gold)
        h1   = hit_at_k(results, gold, 1)
        h5   = hit_at_k(results, gold, 5)
        h10  = hit_at_k(results, gold, 10)

        rr_sum += rr
        hits1  += h1
        hits5  += h5
        hits10 += h10

        if verbose:
            found_rank = next(
                (r["rank"] for r in results if r["chunk_id"] == gold), None
            )
            status = f"rank {found_rank}" if found_rank else "MISS"
            print(f"  [{name}] {status:>8}  {query[:70]}")

        if not h10:
            misses.append(q["question"])

    elapsed = time.perf_counter() - t0

    return {
        "retriever": name,
        "n": n,
        "recall@1":  hits1  / n,
        "recall@5":  hits5  / n,
        "recall@10": hits10 / n,
        "mrr@10":    rr_sum / n,
        "elapsed_s": elapsed,
        "hits5":     hits5,
        "misses":    misses,
    }


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def _print_table(metrics_list: list[dict]) -> None:
    header = (f"{'Retriever':<10}  {'R@1':>6}  {'R@5':>6}  {'Hits@5':>7}  "
              f"{'R@10':>6}  {'MRR@10':>7}  {'ms/q':>6}")
    print()
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        print(
            f"{m['retriever']:<10}  "
            f"{m['recall@1']:>6.1%}  "
            f"{m['recall@5']:>6.1%}  "
            f"{str(m['hits5']) + '/' + str(m['n']):>7}  "
            f"{m['recall@10']:>6.1%}  "
            f"{m['mrr@10']:>7.4f}  "
            f"{1000 * m['elapsed_s'] / m['n']:>6.0f}"
        )
    print()


def _print_misses(metrics_list: list[dict]) -> None:
    for m in metrics_list:
        if m["misses"]:
            print(f"{m['retriever']} misses ({len(m['misses'])}/{m['n']}):")
            for q in m["misses"]:
                print(f"  - {q}")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval eval on eval/questions.json."
    )
    parser.add_argument(
        "--retriever", choices=["bm25", "dense", "hybrid", "rerank", "all"],
        default="all", help="Which retriever(s) to evaluate (default: all)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-query rank / miss detail."
    )
    parser.add_argument(
        "--misses", action="store_true",
        help="Print questions that got no hit in top-10."
    )
    parser.add_argument(
        "--rerank-depth", type=int, default=DEFAULT_DEPTH,
        help=f"Hybrid candidates passed to the reranker (default: {DEFAULT_DEPTH})."
    )
    parser.add_argument(
        "--rerank-model", default=MODEL_NAME,
        help=f"Cross-encoder used by the rerank retriever (default: {MODEL_NAME})."
    )
    args = parser.parse_args()

    if args.rerank_depth < EVAL_K:
        print(f"ERROR: --rerank-depth must be >= {EVAL_K}.", file=sys.stderr)
        sys.exit(2)

    if not QUESTIONS_PATH.exists():
        print(f"ERROR: {QUESTIONS_PATH} not found. Run eval/generate_questions.py first.",
              file=sys.stderr)
        sys.exit(1)

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Evaluating {len(questions)} questions  (k={EVAL_K})")

    retrievers = (
        ["bm25", "dense", "hybrid", "rerank"] if args.retriever == "all"
        else [args.retriever]
    )

    results = []
    for name in retrievers:
        print(f"\nRunning {name}...")
        if args.verbose:
            print()
        m = _run_one(name, questions, verbose=args.verbose,
                     rerank_depth=args.rerank_depth,
                     rerank_model=args.rerank_model)
        results.append(m)
        if not args.verbose:
            print(f"  done in {m['elapsed_s']:.1f}s")

    _print_table(results)

    if args.misses or args.verbose:
        _print_misses(results)


if __name__ == "__main__":
    main()
