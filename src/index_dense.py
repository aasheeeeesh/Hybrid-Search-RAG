# src/index_dense.py
# Phase 3 — Dense Indexing
#
# Loads chunks.json, embeds them with BAAI/bge-small-en-v1.5, and persists
# the vectors + metadata into a Chroma collection on disk at .chroma/.
#
# Usage:
#   python -m src.index_dense --smoke-test   # verify model + one embedding
#   python -m src.index_dense --build        # embed all chunks and write to Chroma
#   python -m src.index_dense --verify       # sanity-check the persisted collection

import argparse
import json
import os
import time

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = os.path.join("data", "processed", "chunks.json")
CHROMA_PATH = ".chroma"
COLLECTION_NAME = "chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32


def build_searchable(chunk: dict) -> str:
    """Same recipe as BM25 indexing: title + heading + body concatenated.

    Kept identical across sparse and dense so downstream (Phase 4 fusion)
    reasons about the same content per chunk_id in both indexes.
    """
    return (
        f"{chunk.get('doc_title', '')}. "
        f"{chunk.get('section_heading', '')}. "
        f"{chunk.get('text', '')}"
    )


def load_chunks(path: str = CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    print(f"  Model ready in {time.perf_counter() - t0:.2f}s")
    dim_fn = getattr(model, "get_embedding_dimension",
                     getattr(model, "get_sentence_embedding_dimension", None))
    print(f"  Embedding dimension: {dim_fn() if dim_fn else 'unknown'}")
    print(f"  Max sequence length (tokens): {model.max_seq_length}")
    return model


def smoke_test(chunks: list[dict], model: SentenceTransformer) -> None:
    """Embed a single chunk and print the first 8 values of the vector."""
    chunk = chunks[0]
    text = build_searchable(chunk)
    print("\n--- Smoke test on chunk 0 ---")
    print(f"chunk_id : {chunk['chunk_id']}")
    print(f"title    : {chunk.get('doc_title', '')}")
    print(f"section  : {chunk.get('section_heading', '')}")
    print(f"tokens   : {chunk['token_count']}")

    t0 = time.perf_counter()
    vector = model.encode(text, normalize_embeddings=True)
    elapsed = time.perf_counter() - t0
    print(f"\nEncoded in {elapsed * 1000:.1f} ms")
    print(f"Vector shape : {vector.shape}")
    print(f"First 8 dims : {vector[:8]}")
    print(f"L2 norm      : {(vector ** 2).sum() ** 0.5:.4f}  (should be ~1.0)")


# ---------------------------------------------------------------------------
# Chroma persistence
# ---------------------------------------------------------------------------


def get_or_reset_collection(client: chromadb.PersistentClient, reset: bool):
    """Return the target collection. When `reset` is True, drop it first so a
    rebuild starts from an empty state (avoids stale vectors from prior runs).

    Cosine similarity is set at collection-creation time via `hnsw:space`.
    """
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  Dropped existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass  # collection did not exist yet
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(
    chunks: list[dict],
    model: SentenceTransformer,
    chroma_path: str = CHROMA_PATH,
) -> None:
    """Encode all chunks and upsert into a fresh Chroma collection."""
    print(f"\nBuilding searchable text for {len(chunks):,} chunks …")
    texts = [build_searchable(c) for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    # Chroma metadata must be scalar (str/int/float/bool) — no nested dicts.
    metadatas = [
        {
            "source_path": c.get("source_path", ""),
            "doc_title": c.get("doc_title", ""),
            "section_heading": c.get("section_heading", ""),
            "token_count": int(c.get("token_count", 0)),
        }
        for c in chunks
    ]

    print(f"Encoding with batch_size={BATCH_SIZE} … (progress bar below)")
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    encode_time = time.perf_counter() - t0
    print(f"  Encoded {len(embeddings):,} vectors in {encode_time:.1f}s "
          f"({len(embeddings) / encode_time:.1f} chunks/sec)")

    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = get_or_reset_collection(client, reset=True)

    print(f"Upserting {len(ids):,} vectors into Chroma at '{chroma_path}' …")
    t1 = time.perf_counter()
    # Chroma has a per-call size limit; add in the same batch size we encoded with.
    for start in range(0, len(ids), BATCH_SIZE * 8):
        end = start + BATCH_SIZE * 8
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
            documents=texts[start:end],
        )
    upsert_time = time.perf_counter() - t1
    print(f"  Upsert done in {upsert_time:.1f}s")

    total = collection.count()
    print(f"\nCollection '{COLLECTION_NAME}' now holds {total:,} items "
          f"(expected {len(chunks):,}).")
    if total != len(chunks):
        print("  WARNING: count mismatch — the index is not what you think it is.")


def verify(chroma_path: str = CHROMA_PATH) -> None:
    """Open the persisted collection and print a few basic sanity numbers."""
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()
    print(f"Collection '{COLLECTION_NAME}' at '{chroma_path}': {total:,} items")

    sample = collection.peek(limit=1)
    if not sample["ids"]:
        print("  Collection is empty.")
        return

    print("\n--- Sample entry ---")
    print(f"id       : {sample['ids'][0]}")
    print(f"metadata : {sample['metadatas'][0]}")
    print(f"document : {sample['documents'][0][:200]} …")
    print(f"embedding: dim={len(sample['embeddings'][0])}, "
          f"first 4 = {sample['embeddings'][0][:4]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense indexing pipeline.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Load model + embed one chunk (fast toolchain check).")
    parser.add_argument("--build", action="store_true",
                        help="Encode all chunks and write to Chroma.")
    parser.add_argument("--verify", action="store_true",
                        help="Open the persisted collection and print stats.")
    parser.add_argument("--chunks", default=CHUNKS_PATH)
    parser.add_argument("--chroma", default=CHROMA_PATH)
    args = parser.parse_args()

    if args.verify and not (args.build or args.smoke_test):
        verify(args.chroma)
    else:
        chunks = load_chunks(args.chunks)
        print(f"Loaded {len(chunks):,} chunks from {args.chunks}")
        model = load_model()

        if args.smoke_test:
            smoke_test(chunks, model)
        if args.build:
            build_index(chunks, model, args.chroma)
        if args.verify:
            print()
            verify(args.chroma)

        if not (args.smoke_test or args.build or args.verify):
            print("\n(Pass --smoke-test, --build, or --verify.)")
