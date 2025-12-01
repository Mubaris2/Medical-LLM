import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("medical_faiss.index")

with open("docs.json") as f:
    documents = json.load(f)

model_name = "../model/final_lora_weights"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def query_rag(question, top_k=3):
    print("\n[1] Encoding query for retrieval...")
    query_embedding = embedder.encode([question])
    print("[2] Searching vector DB...")
    distances, indices = index.search(np.array(query_embedding), top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = "\n\n".join(retrieved_docs)
    prompt = f"""
You are a medical assistant AI. Answer strictly based on the provided medical documents.
If the information is not available, say "I don't know".

### Question:
{question}

### Relevant Documents:
{context}

### Final Answer:"""

    print("[3] Generating answer using LLM...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer, retrieved_docs

q = "What are the symptoms of dengue fever?"
answer, docs = query_rag(q)

print("\n===== Final Answer =====")
print(answer)

print("\n===== Retrieved Docs =====")
for d in docs:
    print("\n---\n", d)
