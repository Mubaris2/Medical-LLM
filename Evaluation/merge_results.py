import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt')

with open("base_results.json") as f:
    base = {x["id"]: x for x in json.load(f)}

with open("lora_results.json") as f:
    lora = {x["id"]: x for x in json.load(f)}

with open("rag_results.json") as f:
    rag = {x["id"]: x for x in json.load(f)}

with open("../DataEngineering/eval_testcases.json") as f:
    testcases = {t["id"]: t for t in json.load(f)}

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def factual_accuracy(pred, keywords):
    if not keywords:
        return 1.0
    hits = sum(1 for k in keywords if k.lower() in pred.lower())
    return hits / len(keywords)


def hallucination_penalty(pred, context):
    if not context:
        return 0.0
    
    pred_words = set(pred.lower().split())
    ctx_words = set(context.lower().split())
    
    hallucinated = pred_words - ctx_words
    penalty = len(hallucinated) / max(len(pred_words), 1)
    
    return penalty


def rag_precision(context, question):
    if not context:
        return 0.0
    
    q_emb = embed_model.encode(question, convert_to_tensor=True)
    ctx_emb = embed_model.encode(context, convert_to_tensor=True)
    
    score = float(util.pytorch_cos_sim(q_emb, ctx_emb))
    return score

final = []

for tc_id in base:

    testcase = testcases[tc_id]
    expected = testcase["expected_keywords"]
    question = testcase["input"]

    base_pred = base[tc_id]["prediction_base"]
    lora_pred = lora[tc_id]["prediction_lora"]
    rag_pred = rag[tc_id]["prediction_rag"]
    rag_context = rag[tc_id].get("used_context", "")

    rouge_base = scorer.score(" ".join(expected), base_pred)["rougeL"].fmeasure
    rouge_lora = scorer.score(" ".join(expected), lora_pred)["rougeL"].fmeasure
    rouge_rag  = scorer.score(" ".join(expected), rag_pred )["rougeL"].fmeasure

    bleu_base = sentence_bleu([expected], base_pred.split())
    bleu_lora = sentence_bleu([expected], lora_pred.split())
    bleu_rag  = sentence_bleu([expected], rag_pred.split())

    emb_base = util.pytorch_cos_sim(
        embed_model.encode(base_pred), 
        embed_model.encode(" ".join(expected))
    ).item()

    emb_lora = util.pytorch_cos_sim(
        embed_model.encode(lora_pred), 
        embed_model.encode(" ".join(expected))
    ).item()

    emb_rag = util.pytorch_cos_sim(
        embed_model.encode(rag_pred), 
        embed_model.encode(" ".join(expected))
    ).item()

    fact_base = factual_accuracy(base_pred, expected)
    fact_lora = factual_accuracy(lora_pred, expected)
    fact_rag  = factual_accuracy(rag_pred, expected)

    hall_base = 0
    hall_lora = 0
    hall_rag  = hallucination_penalty(rag_pred, rag_context)

    rag_prec = rag_precision(rag_context, question)

    final.append({
        "id": tc_id,
        "input": question,

        "base": base_pred,
        "lora": lora_pred,
        "rag": rag_pred,

        "metrics": {
            "rouge_l": {
                "base": rouge_base,
                "lora": rouge_lora,
                "rag": rouge_rag
            },
            "bleu": {
                "base": bleu_base,
                "lora": bleu_lora,
                "rag": bleu_rag
            },
            "embedding_similarity": {
                "base": emb_base,
                "lora": emb_lora,
                "rag": emb_rag
            },
            "factual_accuracy": {
                "base": fact_base,
                "lora": fact_lora,
                "rag": fact_rag
            },
            "hallucination_penalty": {
                "base": hall_base,
                "lora": hall_lora,
                "rag": hall_rag
            },
            "rag_retrieval_precision": rag_prec
        },

        "rag_context_used": rag_context
    })

with open("evaluation_results.json", "w") as f:
    json.dump(final, f, indent=4)
