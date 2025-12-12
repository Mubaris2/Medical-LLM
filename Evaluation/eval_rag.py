import json
from rag_engine import retrieve
from unsloth import FastLanguageModel
import torch

with open("../DataEngineering/eval_testcases.json") as f:
    testcases = json.load(f)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="../model/merged_model",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto",
)
results = []

for tc in testcases:
    retrieved_context = retrieve(tc["input"])

    prompt = f"Use this context to answer:\n{retrieved_context}\n\nQuestion: {tc['input']}"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=150)

    prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    results.append({
        "id": tc["id"],
        "input": tc["input"],
        "prediction_rag": prediction,
        "used_context": retrieved_context
    })

with open("rag_results.json", "w") as f:
    json.dump(results, f, indent=4)
