# eval/eval_verify.py
# Phase 7.5 — Measure the NLI claim checker on hand-labelled claims.
#
# Each claim in eval/claims.json is checked against the chunk it was written
# from. "supported" claims should pass; "contradicted" and "not_stated" claims
# should not. The claims are hand-written, so this measures the checker, not
# the faithfulness of generated answers.
#
# Usage:
#   python -m eval.eval_verify
#   python -m eval.eval_verify --windows 0 1 2 3 --threshold 0.5
#   python -m eval.eval_verify --claims eval/claims_cli.json --k 5

import argparse
import json
from pathlib import Path

from src.rerank import search_reranked
from src.verify import SUPPORT_THRESHOLD, WINDOW_SENTENCES, get_model, verify_claim

CLAIMS_PATH = Path("eval/claims.json")
CHUNKS_PATH = Path("data/processed/chunks.json")
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
KINDS = ["supported", "contradicted", "not_stated"]


def _score_claims(claims, texts, window, model, k=None):
    """Score each claim against its own chunk, or the top *k* reranked chunks like the CLI."""
    for c in claims:
        chunks = ([h["text"] for h in search_reranked(c["claim"], k=k)] if k
                  else [texts[c["chunk_id"]]])
        results = [verify_claim(c["claim"], t, window=window, model=model) for t in chunks]
        yield c, max(r["score"] for r in results)


def _report(scored: list[tuple[dict, float]], threshold: float) -> None:
    print(f"{'Threshold':>9}  {'Supported kept':>14}  {'Contradicted caught':>19}  "
          f"{'Not-stated caught':>17}  {'Precision':>9}")
    for t in sorted({*THRESHOLDS, threshold}):
        counts = {k: [0, 0] for k in KINDS}
        for c, score in scored:
            counts[c["kind"]][0] += score >= t
            counts[c["kind"]][1] += 1
        passed = sum(v[0] for v in counts.values())
        precision = counts["supported"][0] / passed if passed else 0.0
        sup, con, ns = (counts[k] for k in KINDS)
        print(f"{t:>9}  {f'{sup[0]}/{sup[1]}':>14}  "
              f"{f'{con[1] - con[0]}/{con[1]}':>19}  "
              f"{f'{ns[1] - ns[0]}/{ns[1]}':>17}  {precision:>9.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the NLI claim checker.")
    parser.add_argument("--claims", type=Path, default=CLAIMS_PATH,
                        help=f"Labelled claims file (default: {CLAIMS_PATH}).")
    parser.add_argument("--windows", type=int, nargs="+", default=[WINDOW_SENTENCES],
                        help="Sentences per passage to compare; 0 is the whole chunk.")
    parser.add_argument("--threshold", type=float, default=SUPPORT_THRESHOLD,
                        help=f"Threshold to list errors at (default: {SUPPORT_THRESHOLD}).")
    parser.add_argument("--k", type=int, default=None,
                        help="Check against the top k reranked chunks, as the CLI does, "
                             "instead of the chunk each claim was written from.")
    args = parser.parse_args()

    claims = json.loads(args.claims.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    texts = {c["chunk_id"]: c["text"] for c in chunks}
    missing = [c["chunk_id"] for c in claims if c["chunk_id"] not in texts]
    if missing:
        raise ValueError(f"Claims point at chunks that no longer exist: {missing}")

    model = get_model()
    for window in args.windows:
        print(f"\nwindow={window or 'whole chunk'}  ({len(claims)} claims)")
        scored = list(_score_claims(claims, texts, window, model, args.k))
        _report(scored, args.threshold)
        print(f"\nErrors at threshold {args.threshold}:")
        for c, score in scored:
            if (score >= args.threshold) != (c["kind"] == "supported"):
                print(f"  {score:.3f}  {c['kind']:<12}  {c['claim']}")


if __name__ == "__main__":
    main()
