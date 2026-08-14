# src/index_bm25.py
# Phase 2 — BM25 Indexing
#
# Loads data/processed/chunks.json, builds a BM25Okapi index over
#   doc_title + section_heading + text
# and pickles both the index and the chunk metadata list to
#   data/processed/bm25_index.pkl
#
# Usage:
#   python -m src.index_bm25            # build and save
#   python -m src.index_bm25 --verify  # build, save, then run a quick smoke-test query

import json
import os
import pickle
import re
import time
import argparse

from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
# Conservative splitting strategy:
#   • lowercase                        — "Security" == "security"
#   • split on whitespace + small set  — preserve compound tokens like
#     #security-identity-ops and gitlab-com/gl-security while still
#     breaking on sentence-ending punctuation
#   • drop empty tokens
#   • Snowball-stem each token         — "policies" → "polici", matches "policy"
#
# The SAME function is used at index time AND query time — critical for term
# matching consistency.

_stemmer = SnowballStemmer("english")

# Characters we split on (in addition to whitespace).
# Excludes - and _ so hyphenated compound tokens (anti-harassment,
# #security-identity-ops) are preserved as single matchable units.
# Includes / so URL path segments (gitlab.com/gl-security/foo) are split
# into individual tokens rather than one unsearchable mega-token.
_SPLIT_PATTERN = re.compile(r"[\s.,!?;:()\[\]{}<>\"'`|\\=+*&^%$~/]+")


def tokenize(text: str) -> list[str]:
    """Convert a string to a list of stemmed tokens.

    Pipeline:
      1. Lowercase
      2. Split on whitespace + punctuation including /
           - Preserves hyphens (-) → anti-harassment, #security-identity-ops
             remain as single compound tokens for exact-match queries.
           - Splits on / → gitlab.com/gl-security/foo → ['gitlab.com',
             'gl-security', 'foo'], so path-segment queries find results
             even when the term appears inside a longer URL in the corpus.
      3. Drop empty strings
      4. Stem with Snowball English stemmer
    """
    # 1. Lowercase
    text = text.lower()
    # 2. Split on whitespace + minimal punctuation set
    raw_tokens = _SPLIT_PATTERN.split(text)
    # 3 & 4. Drop empties, stem survivors
    return [_stemmer.stem(t) for t in raw_tokens if t]


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

CHUNKS_PATH = os.path.join("data", "processed", "chunks.json")
INDEX_PATH = os.path.join("data", "processed", "bm25_index.pkl")


def build_index(chunks_path: str = CHUNKS_PATH) -> tuple[BM25Okapi, list[dict]]:
    """Load chunks, tokenize, build BM25Okapi index.

    Returns (bm25, chunks) so callers can also retrieve the metadata list.
    """
    print(f"Loading chunks from {chunks_path} …")
    with open(chunks_path, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)
    print(f"  {len(chunks):,} chunks loaded.")

    print("Building tokenisable text and tokenising …")
    t0 = time.perf_counter()
    tokenized_corpus: list[list[str]] = []
    for chunk in chunks:
        # Concatenate heading signals with body — headings get extra weight
        # just by appearing in the searchable text (their terms score twice
        # if the body also mentions them, once otherwise).
        searchable = (
            f"{chunk.get('doc_title', '')}. "
            f"{chunk.get('section_heading', '')}. "
            f"{chunk.get('text', '')}"
        )
        tokenized_corpus.append(tokenize(searchable))
    elapsed_tok = time.perf_counter() - t0
    print(f"  Tokenisation done in {elapsed_tok:.2f}s.")

    print("Building BM25Okapi index (k1=1.5, b=0.75) …")
    t1 = time.perf_counter()
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    elapsed_idx = time.perf_counter() - t1
    print(f"  Index built in {elapsed_idx:.2f}s.")

    return bm25, chunks


def save_index(bm25: BM25Okapi, chunks: list[dict], index_path: str = INDEX_PATH) -> None:
    """Pickle the BM25 object + chunk metadata list to disk."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    payload = {"bm25": bm25, "chunks": chunks}
    with open(index_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(index_path) / 1_048_576
    print(f"Index saved to {index_path}  ({size_mb:.1f} MB).")


def load_index(index_path: str = INDEX_PATH) -> tuple[BM25Okapi, list[dict]]:
    """Load a previously pickled BM25 index from disk."""
    with open(index_path, "rb") as f:
        payload = pickle.load(f)
    return payload["bm25"], payload["chunks"]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save the BM25 index.")
    parser.add_argument(
        "--chunks", default=CHUNKS_PATH, help="Path to chunks.json (default: %(default)s)"
    )
    parser.add_argument(
        "--output", default=INDEX_PATH, help="Where to write the pickle (default: %(default)s)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After building, run a quick smoke-test query and print top-3 results.",
    )
    args = parser.parse_args()

    bm25, chunks = build_index(args.chunks)
    save_index(bm25, chunks, args.output)

    if args.verify:
        print("\n--- Smoke-test query: 'anti-harassment policy' ---")
        from src.retrieve_bm25 import search_bm25  # noqa: E402

        results = search_bm25("anti-harassment policy", k=3, bm25=bm25, chunks=chunks)
        for r in results:
            print(
                f"  [{r['rank']}] score={r['score']:.4f}  {r['chunk_id']}\n"
                f"       {r['doc_title']} | {r['section_heading']}\n"
                f"       {r['text'][:120]} …\n"
            )
