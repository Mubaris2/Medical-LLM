from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

with open("../DataEngineering/eval_testcases.json") as f:
    testcases = json.load(f)

results = []

for tc in testcases:
    input_text = tc["input"]

    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=150)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    results.append({
        "id": tc["id"],
        "input": input_text,
        "prediction_base": prediction
    })

with open("base_results.json", "w") as f:
    json.dump(results, f, indent=4)
