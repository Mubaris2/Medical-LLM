import faiss, json, numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_PATH = "rag/medical_faiss.index"
DOCS_PATH = "rag/docs.json"

_index = None
_documents = None

def load_index_and_docs():
    global _index, _documents
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if _documents is None:
        with open(DOCS_PATH, "r") as f:
            _documents = json.load(f)
    return _index, _documents

_model = None
_tokenizer = None
def load_llm(model_path="../model/final_lora_weights"):
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    return _tokenizer, _model

def embed_text(texts):
    return EMBED_MODEL.encode(texts)
