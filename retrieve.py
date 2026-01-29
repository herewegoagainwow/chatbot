import faiss
import pickle
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

'''Loads a FAISS index and original chunk metadata from disk,
 embeds a sample query using the same HuggingFace model,
 performs a similarity search, and prints the top matching chunks.'''

index = faiss.read_index("text.index")
chunks = pickle.load(open("chunks.pkl", "rb"))

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    encode_kwargs={"normalize_embeddings": True}
)

query = "What power input does the board support?"

q_emb = embeddings.embed_query(query)
q_emb = np.array([q_emb]).astype("float32")  # 🔴 IMPORTANT

D, I = index.search(q_emb, 3)

for idx in I[0]:
    print("PAGE:", chunks[idx]["page"])
    print(chunks[idx]["text"][:300])
    print("-" * 50)
