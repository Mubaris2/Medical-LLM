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

class SymptomRequest(BaseModel):
    symptoms: List[str]
    top_k: Optional[int] = 5

class SymptomPrediction(BaseModel):
    disease: str
    reason: str
    confidence: float
