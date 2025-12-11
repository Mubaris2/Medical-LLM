from fastapi import FastAPI, HTTPException, Depends
from schema import QueryRequest, QueryResponse, DiseaseRequest, SymptomRequest, SymptomPrediction, DiseaseResponse
from deps import load_index_and_docs, embed_text, load_llm
import numpy as np, torch, json, re
from typing import List

app = FastAPI(title="MedicalLLM RAG API")

index, docs = load_index_and_docs()
tokenizer, model = load_llm()

def retrieve_docs(question: str, top_k: int=3):
    q_emb = embed_text([question])
    distances, indices = index.search(np.array(q_emb), top_k)
    retrieved = [docs[i] for i in indices[0] if i < len(docs)]
    return retrieved, distances[0].tolist()

@app.post("/query", response_model=QueryResponse)
def query_rag_endpoint(req: QueryRequest):
    retrieved, dists = retrieve_docs(req.query, req.top_k)
    if not retrieved:
        raise HTTPException(status_code=404, detail="No supporting documents found.")
    context = "\n\n".join(retrieved)
    prompt = f"You are a medical assistant. Use only the following documents to answer. If not present, say 'I don't know'.\n\nQuestion: {req.query}\n\nDocuments:\n{context}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return QueryResponse(answer=answer, retrieved_docs=retrieved)

@app.post("/disease_summary", response_model=DiseaseResponse)
def disease_summary(req: DiseaseRequest):
    disease = req.disease.strip().lower()
    hits = [d for d in docs if d.lower().startswith(disease)]
    if not hits:
        query = f"Provide structured medical information about {req.disease}."
        retrieved_docs, _ = retrieve_docs(query, top_k=3)
        context = "\n\n".join(retrieved_docs)
    else:
        context = hits[0]

    prompt = f"""
You are a medical assistant. Read the document below and extract only the following fields:

- Summary (1-2 lines)
- Cause
- Symptoms (as a list)
- Treatments (as a list)
- Emotional_support (1 comforting sentence)

Return output STRICTLY in JSON format with keys:
summary, cause, symptoms, treatments, emotional_support

Document:
{context}

JSON Output:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.2, do_sample=False)

    raw = tokenizer.decode(out[0], skip_special_tokens=True)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise HTTPException(status_code=500, detail="Model did not return valid JSON.")

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse model's JSON output.")
    return DiseaseResponse(
        disease=req.disease,
        summary=data.get("summary", "Not available"),
        cause=data.get("cause", "Not available"),
        symptoms=data.get("symptoms", []),
        treatments=data.get("treatments", []),
        emotional_support=data.get("emotional_support", "You'll be okay."),
        nearby_hospitals=[]
    )

@app.post("/symptom_checker", response_model=List[SymptomPrediction])
def symptom_checker(req: SymptomRequest):
    symptom_text = " ".join(req.symptoms)
    q_emb = embed_text([symptom_text])
    distances, indices = index.search(np.array(q_emb), req.top_k)
    preds = []
    for idx, dist in zip(indices[0], distances[0]):
        doc = docs[idx] if idx < len(docs) else ""
        disease_name = doc.split(".",1)[0] if "." in doc else "Unknown"
        confidence = float(np.exp(-dist))
        reason = f"Matches symptoms: {', '.join(req.symptoms)} (doc similarity)"
        preds.append(SymptomPrediction(disease=disease_name, reason=reason, confidence=confidence))
    return preds
