from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
lora_path = "../model/lora_weights"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto"
)

model = PeftModel.from_pretrained(model, lora_path)

with open("../DataEngineering/eval_testcases.json") as f:
    testcases = json.load(f)

results = []

for tc in testcases:
    inputs = tokenizer(tc["input"], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=150)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    results.append({
        "id": tc["id"],
        "prediction_lora": prediction
    })

with open("lora_results.json", "w") as f:
    json.dump(results, f, indent=4)
