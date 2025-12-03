from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[str]

class DiseaseRequest(BaseModel):
    disease: str

class DiseaseResponse(BaseModel):
    disease: str
    summary: str
    cause: str
    symptoms: List[str]
    treatments: List[str]
    emotional_support: str
    nearby_hospitals: Optional[List[str]] = None
    
class SymptomRequest(BaseModel):
    symptoms: List[str]
    top_k: Optional[int] = 5

class SymptomPrediction(BaseModel):
    disease: str
    reason: str
    confidence: float
