import os, faiss, json, torch
from sentence_transformers import SentenceTransformer

USE_GPU = os.getenv("USE_GPU", "0") == "1"
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU and HAS_CUDA else "cpu"

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_PATH = "/app/Rag/medical_faiss.index"
DOCS_PATH = "/app/Rag/docs.json"

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
def load_llm(model_path="app/merged_model"):
    global _model, _tokenizer
    if _model is not None:
        return _tokenizer, _model
    if DEVICE == "cuda":
        from unsloth import FastLanguageModel

        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        CPU_MODEL = os.getenv("CPU_MODEL", "sshleifer/tiny-gpt2")

        _tokenizer = AutoTokenizer.from_pretrained(CPU_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            CPU_MODEL,
            dtype=torch.float32,
        )
        _model.to("cpu")
        _model.eval()

    return _tokenizer, _model

def embed_text(texts):
    return EMBED_MODEL.encode(texts)
