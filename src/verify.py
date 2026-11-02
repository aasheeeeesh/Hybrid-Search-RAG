# src/verify.py
# Phase 7.5 — Claim verification with NLI.
#
# A citation can point at a real chunk that doesn't say what the claim says.
# This reads the chunk as the premise and the claim as the hypothesis, and only
# counts the claim as supported if the model finds entailment. Runs locally.
#
# Long chunks are split into short overlapping passages first: see WINDOW_SENTENCES.
#
# Provides:
#   split_passages(text, window)                -> list[str]
#   verify_claim(claim, text, *, ...)           -> dict
#   verify_claims(claims, texts_by_id, *, ...)  -> dict
#
# CLI usage:
#   python -m src.verify "a statement to check against the corpus"
#   python -m src.verify "a statement" --k 10

import argparse
import re
import sys

from sentence_transformers import CrossEncoder

from src.rerank import search_reranked

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
MAX_LENGTH = 512
BATCH_SIZE = 8

# Sentences per passage; 0 reads the whole chunk (truncated at MAX_LENGTH).
# From eval/eval_verify.py: chosen on eval/claims.json, then tested on
# eval/claims_long.json, where at threshold 0.5, 3 kept 7/9 supported claims and
# caught 8/9 unsupported ones. The whole chunk kept 5/9 and caught 9/9.
WINDOW_SENTENCES = 3

# Minimum entailment probability for a claim to count as supported. Chosen on
# eval/claims.json and eval/claims_long.json, then tested on eval/claims_cli.json
# against the top 5 reranked chunks: moving from 0.5 to 0.9 caught 16/19 unsupported
# claims instead of 13/19, and kept 13/15 supported claims either way.
SUPPORT_THRESHOLD = 0.9

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

# ---------------------------------------------------------------------------
# Lazy singleton — same rationale as retrieve_dense.get_model().
# ---------------------------------------------------------------------------

_MODEL: CrossEncoder | None = None


def get_model() -> CrossEncoder:
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)
    return _MODEL


def split_passages(text: str, window: int = WINDOW_SENTENCES) -> list[str]:
    """Split *text* into overlapping runs of *window* sentences."""
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    if not sentences:
        return []
    if window <= 0 or len(sentences) <= window:
        return [" ".join(sentences)]
    return [
        " ".join(sentences[i : i + window])
        for i in range(len(sentences) - window + 1)
    ]


def verify_claim(
    claim: str,
    text: str,
    *,
    window: int = WINDOW_SENTENCES,
    threshold: float = SUPPORT_THRESHOLD,
    model: CrossEncoder | None = None,
) -> dict:
    """Check whether *text* supports *claim*.

    Returns `supported`, `score` (the best entailment probability across the
    passages) and `evidence` (the passage that scored it).
    """
    passages = split_passages(text, window)
    if not passages:
        return {"supported": False, "score": 0.0, "evidence": ""}

    model = model or get_model()
    labels = {v.lower(): k for k, v in model.model.config.id2label.items()}
    probs = model.predict(
        [(p, claim) for p in passages],
        apply_softmax=True,
        batch_size=BATCH_SIZE,
    )
    scores = [float(row[labels["entailment"]]) for row in probs]
    best = max(range(len(scores)), key=lambda i: scores[i])
    return {
        "supported": scores[best] >= threshold,
        "score": scores[best],
        "evidence": passages[best],
    }


def verify_claims(
    claims: list[dict],
    texts_by_id: dict[int, str],
    *,
    window: int = WINDOW_SENTENCES,
    threshold: float = SUPPORT_THRESHOLD,
    model: CrossEncoder | None = None,
) -> dict:
    """Split claims ({"text", "citations"}) by whether a cited text supports them.

    A claim needs one supporting citation. Citations missing from *texts_by_id*
    can't support anything. `faithfulness` is the supported share of claims, or
    None when there are no claims to measure.
    """
    supported, unsupported = [], []
    for claim in claims:
        best = {"supported": False, "score": 0.0, "evidence": "", "citation": None}
        for citation in claim["citations"]:
            if citation not in texts_by_id:
                continue
            result = verify_claim(claim["text"], texts_by_id[citation], window=window,
                                  threshold=threshold, model=model)
            if result["score"] >= best["score"]:
                best = {**result, "citation": citation}
        (supported if best["supported"] else unsupported).append({**claim, **best})

    return {
        "supported": supported,
        "unsupported": unsupported,
        "faithfulness": len(supported) / len(claims) if claims else None,
    }


# ---------------------------------------------------------------------------
# CLI pretty-printer
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check a statement against the top reranked chunks."
    )
    parser.add_argument("claim", help="Statement to check.")
    parser.add_argument("--k", type=int, default=5,
                        help="Chunks to check against (default: 5).")
    args = parser.parse_args()

    hits = search_reranked(args.claim, k=args.k)
    if not hits:
        print(f'No results returned for "{args.claim}".', file=sys.stderr)
        sys.exit(1)

    results = [(hit, verify_claim(args.claim, hit["text"])) for hit in hits]
    results.sort(key=lambda pair: pair[1]["score"], reverse=True)

    print(f'\nClaim: "{args.claim}"')
    print("=" * 70)
    print("Supported" if results[0][1]["supported"] else "No supporting passage found")
    for hit, result in results:
        print(f"\n[{result['score']:.3f}] {hit['chunk_id']} §{hit['section_heading']}")
        print(f"     {result['evidence'][:200]}")
