from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("../data/MedicalLLM_final_dataset.json") as f:
    data = json.load(f)

docs = []
for d in data:
    docs.append(
        f"{d['disease']}. {d['summary']} {d['cause']} {d['treatments']}"
    )

embeddings = model.encode(docs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "medical_faiss.index")

with open("docs.json", "w") as f:
    json.dump(docs, f)
