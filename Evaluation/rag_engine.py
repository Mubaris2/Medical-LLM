import faiss
import json
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("../rag/medical_faiss.index")

with open("../rag/docs.json", "r") as f:
    docs = json.load(f)

def retrieve(query, k=3):
    query_emb = embedder.encode([query]).astype("float32")

    distances, indices = index.search(query_emb, k)

    retrieved_docs = []
    for idx in indices[0]:
        retrieved_docs.append(docs[idx])

    return "\n\n".join(retrieved_docs)
