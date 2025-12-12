import faiss, json, torch
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_PATH = "../rag/medical_faiss.index"
DOCS_PATH = "../rag/docs.json"

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
def load_llm(model_path="../model/merged_model"):
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )
    return _tokenizer, _model

def embed_text(texts):
    return EMBED_MODEL.encode(texts)
