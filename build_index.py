import os
import json
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

TEXT_DIR = "data/pdf_name/text"
OUT_DIR = "rag_store"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

CHUNK_SIZE = 450      # tokens-ish (approx via chars)
CHUNK_OVERLAP = 80

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

embedder = SentenceTransformer(EMBED_MODEL)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

# ============================================================
# UTILS
# ============================================================

def split_sections(text: str):
    sections = {}
    current = None
    buffer = []

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("---") and line.endswith("---"):
            if current and buffer:
                sections[current] = "\n".join(buffer).strip()
            current = line.replace("-", "").strip().lower()
            buffer = []
        else:
            buffer.append(line)

    if current and buffer:
        sections[current] = "\n".join(buffer).strip()

    return sections


def chunk_text(text: str):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP

    return chunks

# ============================================================
# STEP 1 — PARSE + CHUNK
# ============================================================

all_chunks = []
chunk_id = 0

for fname in sorted(os.listdir(TEXT_DIR)):
    if not fname.endswith(".txt"):
        continue

    page_num = int(re.findall(r"\d+", fname)[0])
    path = os.path.join(TEXT_DIR, fname)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = split_sections(text)

    for modality, content in sections.items():
        if not content.strip():
            continue

        chunks = chunk_text(content)

        for i, ch in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"c{chunk_id}",
                "page": page_num,
                "modality": modality,   # raw text / tables / diagram context
                "text": ch
            })
            chunk_id += 1

print(f"[OK] Total chunks: {len(all_chunks)}")

# ============================================================
# STEP 2 — EMBED
# ============================================================

texts = [c["text"] for c in all_chunks]
embeddings = embedder.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

# ============================================================
# STEP 3 — FAISS INDEX
# ============================================================

index = faiss.IndexFlatIP(EMBED_DIM)
index.add(embeddings)

# ============================================================
# STEP 4 — SAVE EVERYTHING
# ============================================================

faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))

with open(os.path.join(OUT_DIR, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print("[DONE] RAG index built successfully")
