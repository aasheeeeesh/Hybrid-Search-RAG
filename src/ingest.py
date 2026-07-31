# src/ingest.py
# Phase 1 — Ingestion & Chunking
# Walks data/raw/, cleans .md files, splits into heading-aware chunks,
# attaches metadata, and writes data/processed/chunks.json.
import os
import re
import frontmatter

def clean_body(text):
    # Strip Hugo % shortcodes: {{% ... %}}
    text = re.sub(r'\{\{%.*?%\}\}', '', text, flags=re.DOTALL)
    # Strip Hugo < shortcodes: {{< ... >}}
    text = re.sub(r'\{\{<.*?>\}\}', '', text, flags=re.DOTALL)
    return text

def load_document(path, raw_dir):
    post = frontmatter.load(path)
    title = post.metadata.get("title", "")
    body = clean_body(post.content)
    source_path = os.path.relpath(path, start=raw_dir).replace("\\", "/")
    return {
        "title": title,
        "body": body,
        "source_path": source_path
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest markdown documents")
    parser.add_argument("--input", default="data/raw", help="Path to raw markdown directory")
    parser.add_argument("--output", default="data/processed/chunks.json", help="Path to output chunks JSON")
    args = parser.parse_args()
    
    # For testing Step 1, print a single file
    test_file = os.path.join(args.input, "people-group/anti-harassment.md")
    if os.path.exists(test_file):
        doc = load_document(test_file, args.input)
        print("=== STEP 1 TEST ===")
        print(f"Title: {doc['title']}")
        print(f"Source Path: {doc['source_path']}")
        print(f"Body (first 200 chars):\n{doc['body'][:200]}")
    else:
        print(f"Test file {test_file} not found")