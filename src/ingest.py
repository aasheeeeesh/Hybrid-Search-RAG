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

import tiktoken

def get_parent_h2(heading_path):
    if not heading_path:
        return ""
    return heading_path.split(" > ")[0]

def is_table(paragraph):
    lines = paragraph.split("\n")
    if len(lines) < 2:
        return False
    # A markdown table has pipes in the first 2 lines or a separator line
    has_pipe = all('|' in line for line in lines[:2])
    has_separator = any(re.match(r'^\s*\|?\s*:?-+:?\s*\|(\s*:?-+:?\s*\|)*\s*$', line) for line in lines)
    return has_pipe or has_separator

def split_paragraph_into_sentences(paragraph):
    if is_table(paragraph):
        return [paragraph]
    # Split by punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    return [s.strip() for s in sentences if s.strip()]

def merge_tiny_sections(sections, enc):
    # Calculate token counts
    for s in sections:
        s["token_count"] = len(enc.encode(s["text"]))
        
    # Pass 1: Merge forward (stub merged into next section under same H2)
    merged_1 = []
    i = 0
    n = len(sections)
    while i < n:
        curr = sections[i]
        if curr["token_count"] < 100 and i + 1 < n:
            nxt = sections[i + 1]
            if get_parent_h2(curr["heading"]) == get_parent_h2(nxt["heading"]):
                nxt["text"] = curr["text"] + "\n\n" + nxt["text"]
                nxt["token_count"] = len(enc.encode(nxt["text"]))
                i += 1
                continue
        merged_1.append(curr)
        i += 1
        
    # Pass 2: Merge backward (remaining stubs merged into previous section under same H2)
    merged_2 = []
    for s in merged_1:
        if s["token_count"] < 100 and merged_2:
            prev = merged_2[-1]
            if get_parent_h2(s["heading"]) == get_parent_h2(prev["heading"]):
                prev["text"] = prev["text"] + "\n\n" + s["text"]
                prev["token_count"] = len(enc.encode(prev["text"]))
                continue
        merged_2.append(s)
        
    return merged_2

def assemble_chunks(units, enc):
    chunks = []
    current_chunk_units = []
    current_tokens = 0
    
    for unit in unit_tokens_list(units, enc):
        unit_text, unit_tokens = unit
        if current_chunk_units and current_tokens + unit_tokens > 500:
            chunks.append("\n\n".join(current_chunk_units))
            current_chunk_units = [unit_text]
            current_tokens = unit_tokens
        else:
            current_chunk_units.append(unit_text)
            current_tokens += unit_tokens
            
    if current_chunk_units:
        chunks.append("\n\n".join(current_chunk_units))
    return chunks

def unit_tokens_list(units, enc):
    return [(unit, len(enc.encode(unit))) for unit in units]

def post_process_chunks(chunks, enc):
    if len(chunks) <= 1:
        return chunks
    
    processed = []
    i = 0
    while i < len(chunks):
        curr = chunks[i]
        curr_tokens = len(enc.encode(curr))
        if curr_tokens < 150 and i + 1 < len(chunks):
            nxt = chunks[i+1]
            nxt_tokens = len(enc.encode(nxt))
            if curr_tokens + nxt_tokens <= 550:
                chunks[i+1] = curr + "\n\n" + nxt
                i += 1
                continue
        elif curr_tokens < 150 and processed:
            prev = processed[-1]
            prev_tokens = len(enc.encode(prev))
            if curr_tokens + prev_tokens <= 550:
                processed[-1] = prev + "\n\n" + curr
                i += 1
                continue
        processed.append(curr)
        i += 1
    return processed

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

def process_document(doc, enc):
    sections = split_into_sections(doc["body"])
    merged_sections = merge_tiny_sections(sections, enc)
    
    doc_chunks = []
    for sec in merged_sections:
        if sec["token_count"] <= 500:
            doc_chunks.append({
                "section_heading": sec["heading"],
                "text": sec["text"]
            })
        else:
            paragraphs = [p.strip() for p in sec["text"].split("\n\n") if p.strip()]
            units = []
            for p in paragraphs:
                if is_table(p):
                    units.append(p)
                else:
                    p_tokens = len(enc.encode(p))
                    if p_tokens > 400:
                        units.extend(split_paragraph_into_sentences(p))
                    else:
                        units.append(p)
            
            sub_chunks = assemble_chunks(units, enc)
            sub_chunks = post_process_chunks(sub_chunks, enc)
            
            for sc in sub_chunks:
                doc_chunks.append({
                    "section_heading": sec["heading"],
                    "text": sc
                })
                
    final_chunks = []
    for idx, chunk in enumerate(doc_chunks):
        chunk_id = f"{doc['source_path']}::{idx:04d}"
        tokens = len(enc.encode(chunk["text"]))
        final_chunks.append({
            "chunk_id": chunk_id,
            "source_path": doc["source_path"],
            "doc_title": doc["title"],
            "section_heading": chunk["section_heading"],
            "text": chunk["text"],
            "token_count": tokens
        })
    return final_chunks

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Ingest markdown documents")
    parser.add_argument("--input", default="data/raw", help="Path to raw markdown directory")
    parser.add_argument("--output", default="data/processed/chunks.json", help="Path to output chunks JSON")
    args = parser.parse_args()
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    all_chunks = []
    processed_count = 0
    
    for root, _, files in os.walk(args.input):
        for f in files:
            if f.endswith(".md"):
                file_path = os.path.join(root, f)
                try:
                    doc = load_document(file_path, args.input)
                    chunks = process_document(doc, enc)
                    all_chunks.extend(chunks)
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(all_chunks, out_f, indent=2, ensure_ascii=False)
        
    sizes = [c["token_count"] for c in all_chunks]
    h_100 = sum(1 for s in sizes if s < 100)
    h_100_299 = sum(1 for s in sizes if 100 <= s < 300)
    h_300_500 = sum(1 for s in sizes if 300 <= s <= 500)
    h_500 = sum(1 for s in sizes if s > 500)
    
    print("=== CORPUS INGESTION COMPLETE ===")
    print(f"Total files processed: {processed_count}")
    print(f"Total chunks produced: {len(all_chunks)}")
    print("Chunk-count histogram:")
    print(f"  <100 tokens:    {h_100}")
    print(f"  100-299 tokens: {h_100_299}")
    print(f"  300-500 tokens: {h_300_500}")
    print(f"  >500 tokens:    {h_500}")