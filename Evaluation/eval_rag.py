import json
from rag_engine import retrieve
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
lora_path = "../model/lora_weights"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, lora_path)

with open("../DataEngineering/eval_testcases.json") as f:
    testcases = json.load(f)

results = []

for tc in testcases:
    retrieved_context = retrieve(tc["input"])

    prompt = f"Use this context to answer:\n{retrieved_context}\n\nQuestion: {tc['input']}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)

    prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    results.append({
        "id": tc["id"],
        "prediction_rag": prediction,
        "used_context": retrieved_context
    })

with open("rag_results.json", "w") as f:
    json.dump(results, f, indent=4)
