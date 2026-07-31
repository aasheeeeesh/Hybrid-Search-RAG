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
    # Collapse runs of 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def split_into_sections(body):
    sections = []
    current_h2 = ""
    current_h3 = ""
    
    lines = body.split("\n")
    current_section_lines = []
    
    def save_section():
        if current_section_lines:
            # Form heading path
            heading_path = ""
            if current_h2:
                heading_path = current_h2
                if current_h3:
                    heading_path = f"{current_h2} > {current_h3}"
            
            sections.append({
                "heading": heading_path,
                "text": "\n".join(current_section_lines).strip()
            })
            current_section_lines.clear()

    for line in lines:
        if line.startswith("## "):
            save_section()
            current_h2 = line[3:].strip().strip('#').strip()
            current_h3 = ""
        elif line.startswith("### "):
            save_section()
            current_h3 = line[4:].strip().strip('#').strip()
        else:
            current_section_lines.append(line)
            
    save_section()
    
    # Filter out empty sections
    return [s for s in sections if s["text"]]

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
    
    # For testing Step 3, print split sections
    test_file = os.path.join(args.input, "people-group/anti-harassment.md")
    if os.path.exists(test_file):
        doc = load_document(test_file, args.input)
        sections = split_into_sections(doc["body"])
        print("=== STEP 3 TEST: SPLIT SECTIONS ===")
        print(f"File: {doc['source_path']}")
        print(f"Total sections found: {len(sections)}")
        for idx, sec in enumerate(sections[:5]):
            print(f"\nSection {idx+1}:")
            print(f"  Heading Path: {sec['heading']!r}")
            print(f"  Snippet (100 chars): {sec['text'][:100]!r}")
    else:
        print(f"Test file {test_file} not found")