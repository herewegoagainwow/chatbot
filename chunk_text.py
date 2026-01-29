import os
import json

'''Reads extracted text files from a PDF, splits 
them into sufficiently long paragraphs, and saves 
each paragraph as a chunk with source page metadata
into a JSON file for downstream indexing or retrieval.
'''

TEXT_DIR = "data/pdf_name/text"
OUT_PATH = "chunks.json"

chunks = []

for file in sorted(os.listdir(TEXT_DIR)):
    with open(os.path.join(TEXT_DIR, file), "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]

    for p in paragraphs:
        chunks.append({
            "page": file,
            "text": p
        })

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Total chunks saved: {len(chunks)}")
