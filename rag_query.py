import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess

# ============================================================
# CONFIG
# ============================================================

RAG_DIR = "rag_store"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 5

OLLAMA_MODEL = "llama3.1:8b"  # change if needed

# ============================================================
# LOAD (CPU ONLY FOR EMBEDDINGS)
# ============================================================

embedder = SentenceTransformer(
    EMBED_MODEL,
    device="cpu"          # <<< IMPORTANT
)

index = faiss.read_index(f"{RAG_DIR}/index.faiss")

with open(f"{RAG_DIR}/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ============================================================
# RETRIEVE
# ============================================================

def retrieve(query, k=TOP_K):
    q_emb = embedder.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q_emb, k)

    results = []
    for i in ids[0]:
        c = chunks[i]
        results.append(c)

    return results

# ============================================================
# PROMPT ASSEMBLY
# ============================================================

def build_prompt(query, contexts):
    ctx_blocks = []

    for c in contexts:
        block = (
            f"[Page {c['page']} | {c['modality']}]\n"
            f"{c['text']}"
        )
        ctx_blocks.append(block)

    context = "\n\n---\n\n".join(ctx_blocks)

    prompt = f"""
You are a technical assistant answering questions using provided documentation.

Use ONLY the context below.
If the answer is not present, say "Not found in the document".

### CONTEXT
{context}

### QUESTION
{query}

### ANSWER
"""
    return prompt.strip()

# ============================================================
# LLM CALL (OLLAMA)
# ============================================================

def ask_llama(prompt):
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")

# ============================================================
# MAIN LOOP
# ============================================================

if __name__ == "__main__":
    print("RAG ready. Type 'exit' to quit.")

    while True:
        query = input("\n>> ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        contexts = retrieve(query)
        prompt = build_prompt(query, contexts)
        answer = ask_llama(prompt)

        print("\n" + answer)
