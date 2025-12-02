from fastapi import FastAPI, HTTPException, Depends
from api.schema import QueryRequest, QueryResponse, DiseaseRequest, SymptomRequest, SymptomPrediction
from api.deps import load_index_and_docs, embed_text, load_llm, _model, _tokenizer
import numpy as np, torch, json
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

@app.post("/disease_summary")
def disease_summary(req: DiseaseRequest):
    disease = req.disease.lower()
    hits = [d for d in docs if d.lower().startswith(disease)]
    if not hits:
        query = f"Provide a concise structured summary for disease: {req.disease}. Include summary, cause, symptoms, treatments, emotional message."
        answer, _ = query_rag_endpoint(QueryRequest(query=query, top_k=3))
        return {"disease": req.disease, "structured": answer.answer if isinstance(answer, QueryResponse) else answer}
    prompt = f"Based strictly on the document below, produce output separated by newlines exactly in this format:\nSummary:\nCause:\nSymptoms:\nTreatments:\nEmotional Support:\n\nDocument:\n{hits[0]}\n\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    if "Summary:" in response:
        response = response.split("Summary:",1)[1].strip()
    return {"disease": req.disease, "structured": response}

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
