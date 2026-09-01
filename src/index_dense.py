# src/index_dense.py
# Phase 3 — Dense Indexing (Step 1: model smoke-test)
#
# Loads chunks.json and the BGE embedding model, then embeds one chunk
# to confirm the toolchain works end-to-end before we spend minutes
# encoding the full corpus.
#
# Usage:
#   python -m src.index_dense --smoke-test

import argparse
import json
import os
import time

from sentence_transformers import SentenceTransformer

CHUNKS_PATH = os.path.join("data", "processed", "chunks.json")
MODEL_NAME = "BAAI/bge-small-en-v1.5"


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
    """First call downloads ~130 MB from HuggingFace into ~/.cache/huggingface/.
    Subsequent calls load from cache in a couple of seconds.
    """
    print(f"Loading embedding model: {model_name}")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    print(f"  Model ready in {time.perf_counter() - t0:.2f}s")
    # sentence-transformers renamed this method; keep a fallback for older versions.
    dim_fn = getattr(model, "get_embedding_dimension",
                     getattr(model, "get_sentence_embedding_dimension", None))
    print(f"  Embedding dimension: {dim_fn() if dim_fn else 'unknown'}")
    print(f"  Max sequence length (tokens): {model.max_seq_length}")
    return model


def smoke_test(chunks: list[dict], model: SentenceTransformer) -> None:
    """Embed a single chunk and print the first 8 values of the vector.
    A short vector print is enough to confirm the encode path works.
    """
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense indexing pipeline.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load model + embed one chunk to verify the toolchain works.",
    )
    parser.add_argument("--chunks", default=CHUNKS_PATH)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks):,} chunks from {args.chunks}")

    model = load_model()

    if args.smoke_test:
        smoke_test(chunks, model)
    else:
        print("\n(Nothing to do — pass --smoke-test to verify the pipeline.)")
