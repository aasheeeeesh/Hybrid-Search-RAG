# src/retrieve_dense.py
# Phase 3 — Dense Retrieval
#
# Provides:
#   search_dense(query, k, *, collection, model) -> list[dict]  ← importable
#
# CLI usage:
#   python -m src.retrieve_dense "your query"
#   python -m src.retrieve_dense "your query" --k 10

import argparse
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from src.index_dense import (
    CHROMA_PATH,
    COLLECTION_NAME,
    MODEL_NAME,
)

# ---------------------------------------------------------------------------
# Lazy singletons — the model and Chroma client are expensive to instantiate.
# Cache them at module scope so a program calling search_dense() in a loop
# (Phase 5's eval driver) pays the cost only once.
# ---------------------------------------------------------------------------

_MODEL: SentenceTransformer | None = None
_COLLECTION: chromadb.api.models.Collection.Collection | None = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def get_collection(chroma_path: str = CHROMA_PATH):
    global _COLLECTION
    if _COLLECTION is None:
        client = chromadb.PersistentClient(path=chroma_path)
        _COLLECTION = client.get_collection(COLLECTION_NAME)
    return _COLLECTION


# ---------------------------------------------------------------------------
# Core retrieval function — same shape as search_bm25()
# ---------------------------------------------------------------------------


def search_dense(
    query: str,
    k: int = 5,
    *,
    collection=None,
    model: SentenceTransformer | None = None,
    chroma_path: str = CHROMA_PATH,
) -> list[dict]:
    """Return the top-k dense-retrieval results for *query*.

    Parameters
    ----------
    query        : Natural-language query string.
    k            : Number of results to return.
    collection   : Pre-loaded Chroma collection (optional — avoids repeated
                   client construction in tight loops).
    model        : Pre-loaded SentenceTransformer (optional — avoids reloading
                   the 130 MB model per call).
    chroma_path  : Path to the persisted Chroma store (used when `collection`
                   is None).

    Returns
    -------
    List of dicts, one per result, sorted by descending similarity:
        rank              int    (1-indexed)
        score             float  (cosine similarity, in [0, 1] approx)
        chunk_id          str
        token_count       int
        doc_title         str
        section_heading   str
        text              str
    """
    model = model or get_model()
    collection = collection or get_collection(chroma_path)

    query_vec = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    result = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["metadatas", "documents", "distances"],
    )

    ids = result["ids"][0]
    metadatas = result["metadatas"][0]
    documents = result["documents"][0]
    distances = result["distances"][0]

    hits = []
    for rank, (cid, meta, doc, dist) in enumerate(
        zip(ids, metadatas, documents, distances), start=1
    ):
        # Chroma returns *distance*. With hnsw:space=cosine and unit-norm
        # vectors, distance ≈ 1 - cosine_similarity, so similarity = 1 - dist.
        similarity = float(1.0 - dist)
        hits.append(
            {
                "rank": rank,
                "score": similarity,
                "chunk_id": cid,
                "token_count": int(meta.get("token_count", 0)),
                "doc_title": meta.get("doc_title", ""),
                "section_heading": meta.get("section_heading", ""),
                # Chroma stored the concatenated title+heading+body as the
                # document; strip the "title. heading. " prefix so the printed
                # text matches what BM25's CLI shows.
                "text": _strip_searchable_prefix(doc, meta),
            }
        )
    return hits


def _strip_searchable_prefix(document: str, meta: dict) -> str:
    """Reverse the `f"{title}. {section}. {text}"` concatenation applied at
    index time so the returned `text` field matches the original chunk body."""
    prefix = f"{meta.get('doc_title', '')}. {meta.get('section_heading', '')}. "
    if document.startswith(prefix):
        return document[len(prefix):]
    return document  # defensive: return raw if prefix drifted


# ---------------------------------------------------------------------------
# CLI pretty-printer (same shape as retrieve_bm25 so side-by-side comparison
# in Step 4 is trivial)
# ---------------------------------------------------------------------------

TEXT_PREVIEW_CHARS = 200


def _print_results(results: list[dict], query: str) -> None:
    print(f'\nDense results for: "{query}"')
    print("=" * 70)
    for r in results:
        preview = r["text"][:TEXT_PREVIEW_CHARS]
        if len(r["text"]) > TEXT_PREVIEW_CHARS:
            preview += " ..."
        print(
            f"[{r['rank']}] sim={r['score']:.4f}  "
            f"tokens={r['token_count']}  id={r['chunk_id']}"
        )
        print(f"     Title  : {r['doc_title']}")
        print(f"     Section: {r['section_heading']}")
        print(f"     Text   : {preview}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the dense (Chroma) index and print the top-k results."
    )
    parser.add_argument("query", help="Search query string.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of results to return (default: 5).")
    parser.add_argument("--chroma", default=CHROMA_PATH,
                        help="Path to the persisted Chroma store.")
    args = parser.parse_args()

    results = search_dense(args.query, k=args.k, chroma_path=args.chroma)

    if not results:
        print(f'No results returned for "{args.query}".', file=sys.stderr)
        sys.exit(1)

    _print_results(results, args.query)
