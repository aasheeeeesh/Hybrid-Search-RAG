# eval/calibrate_gate.py
# Phase 7 — Calibrate the low-retrieval-confidence gate.
#
# The reranker's score is a raw logit with no fixed scale, so the "I don't
# know" threshold has to be measured. This compares the top reranked score of
# the 42 answerable gold questions against eval/unanswerable.json, whose topics
# are checked to be absent from the corpus. No API calls: retrieval only.
#
# Usage:
#   python -m eval.calibrate_gate

import json
from pathlib import Path

from src.rerank import search_reranked

QUESTIONS_PATH = Path("eval/questions.json")
UNANSWERABLE_PATH = Path("eval/unanswerable.json")
RAW_DIR = Path("data/raw")
THRESHOLDS = [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]


def _check_absent(unanswerable: list[dict]) -> None:
    """Fail loudly if a corpus update makes an 'unanswerable' topic answerable."""
    corpus = [p.read_text(encoding="utf-8", errors="replace").lower()
              for p in RAW_DIR.rglob("*.md")]
    for q in unanswerable:
        for term in q["absent_terms"]:
            if any(term in text for text in corpus):
                raise ValueError(f"{term!r} appears in the corpus: {q['question']}")


def _top_score(question: str) -> float:
    return search_reranked(question, k=1)[0]["score"]


def main() -> None:
    answerable = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    unanswerable = json.loads(UNANSWERABLE_PATH.read_text(encoding="utf-8"))
    _check_absent(unanswerable)

    gold_scores = sorted(_top_score(q["question"]) for q in answerable)
    groups = {"off_topic": [], "near_miss": []}
    print("Unanswerable questions (top reranked score):")
    for q in unanswerable:
        score = _top_score(q["question"])
        groups[q["kind"]].append(score)
        print(f"  {score:>7.2f}  {q['kind']:<9}  {q['question']}")

    print(f"\nAnswerable: lowest five top scores {[round(s, 2) for s in gold_scores[:5]]}, "
          f"median {gold_scores[len(gold_scores) // 2]:.2f}")

    n_off, n_near = len(groups["off_topic"]), len(groups["near_miss"])
    print(f"\n{'Threshold':>9}  {'Answerable blocked':>18}  "
          f"{'Off-topic blocked':>17}  {'Near-miss blocked':>17}")
    for t in THRESHOLDS:
        blocked = sum(s < t for s in gold_scores)
        off = sum(s < t for s in groups["off_topic"])
        near = sum(s < t for s in groups["near_miss"])
        print(f"{t:>9}  {f'{blocked}/{len(gold_scores)}':>18}  "
              f"{f'{off}/{n_off}':>17}  {f'{near}/{n_near}':>17}")


if __name__ == "__main__":
    main()
