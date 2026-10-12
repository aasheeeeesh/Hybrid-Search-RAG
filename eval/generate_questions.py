# eval/generate_questions.py
# Phase 5 — Gold Q&A generation for the evaluation harness.
#
# Samples diverse chunks from data/processed/chunks.json and formulates one
# realistic natural-language question per chunk using heading/title heuristics.
# Writes eval/questions.json with the schema expected by run_eval.py.
#
# Run once:
#   python -m eval.generate_questions
#   # → prints the 40 questions so you can review / edit questions.json

import json
import random
import re
import sys
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.json")
OUT_PATH = Path("eval/questions.json")

# Target counts per corpus folder (roughly proportional to chunk counts)
FOLDER_TARGETS = {
    "security": 22,
    "people-group": 14,
    "people-policies": 6,
}

MIN_TOKENS = 150   # skip stubs — too sparse to form a meaningful question
RANDOM_SEED = 7


# ---------------------------------------------------------------------------
# Question heuristics
# ---------------------------------------------------------------------------

def _clean(s: str) -> str:
    """Strip leading breadcrumb ('>') segments, lowercase."""
    parts = [p.strip() for p in s.split(">")]
    return parts[-1].strip()


def _question_from_heading(heading: str, title: str) -> str:
    """Formulate a search-engine-style question from a heading + doc title."""
    h = _clean(heading)
    t = title.strip()

    # Heading already looks like a question
    if h.endswith("?"):
        return h

    # Imperatives / how-to patterns
    m = re.match(
        r"^(How to|How do I|How can I|Steps to|Guide to|Process for)\s+(.+)",
        h, re.IGNORECASE
    )
    if m:
        return f"How do I {m.group(2).rstrip('.')}?"

    # "Request X" / "Submit X" / "Apply for X"
    m = re.match(
        r"^(Request|Submit|Apply for|Complete|Create|Configure|Enable|Disable|Remove|Rotate|Update|Manage)\s+(.+)",
        h, re.IGNORECASE
    )
    if m:
        verb = m.group(1).lower()
        obj  = m.group(2).rstrip('.')
        return f"How do I {verb} {obj}?"

    # "X Policy / Guidelines / Procedure / Program"
    m = re.match(
        r"^(.+?)\s+(Policy|Guidelines?|Procedure|Program|Process|Framework|Overview|Requirements?)$",
        h, re.IGNORECASE
    )
    if m:
        return f"What is the {h}?"

    # Heading is just a noun phrase — ask "what does <doc> say about <heading>"
    if len(h) < 60 and h == h.title():
        return f"What does GitLab say about {h}?"

    # Fallback: use the doc title for context
    return f"What is the {t} policy on {h}?" if len(h) < 50 else f"Tell me about {h}."


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_diverse(chunks: list[dict], seed: int = RANDOM_SEED) -> list[dict]:
    """
    One chunk per source_path (no two questions from the same file), stratified
    by top-level folder according to FOLDER_TARGETS.
    """
    rng = random.Random(seed)

    by_folder: dict[str, dict[str, list]] = {}  # folder → {source_path → [chunks]}
    for c in chunks:
        if c["token_count"] < MIN_TOKENS:
            continue
        folder = c["source_path"].split("/")[0]
        if folder not in FOLDER_TARGETS:
            continue
        by_folder.setdefault(folder, {})
        by_folder[folder].setdefault(c["source_path"], [])
        by_folder[folder][c["source_path"]].append(c)

    selected = []
    for folder, target in FOLDER_TARGETS.items():
        paths = list(by_folder.get(folder, {}).keys())
        rng.shuffle(paths)
        count = 0
        for path in paths:
            if count >= target:
                break
            chunk = rng.choice(by_folder[folder][path])
            selected.append(chunk)
            count += 1

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CHUNKS_PATH.exists():
        print(f"ERROR: {CHUNKS_PATH} not found. Run python -m src.ingest first.",
              file=sys.stderr)
        sys.exit(1)

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    sample = _sample_diverse(chunks)
    print(f"Sampled {len(sample)} chunks across corpus.\n")

    questions = []
    for c in sample:
        q = _question_from_heading(c["section_heading"], c["doc_title"])
        entry = {
            "question": q,
            "gold_chunk_id": c["chunk_id"],
            "doc_title": c["doc_title"],
            "section_heading": c["section_heading"],
            "source_path": c["source_path"],
            "token_count": c["token_count"],
        }
        questions.append(entry)
        print(f"  [{c['source_path'].split('/')[0]}] {q}")
        print(f"       -> {c['chunk_id']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(questions)} questions to {OUT_PATH}")
    print("Review and edit questions.json before running eval.")


if __name__ == "__main__":
    main()
