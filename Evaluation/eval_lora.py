from unsloth import FastLanguageModel
import torch
import json

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
lora_path = "../model/final_lora_weights"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model,
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
)

model.load_adapter(lora_path)
model.save_pretrained("merged_model", tokenizer)

with open("../DataEngineering/eval_testcases.json") as f:
    testcases = json.load(f)

del model
del tokenizer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "merged_model",
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
    device_map="auto",
)

results = []

for tc in testcases:
    input_text = tc["input"]

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=150,
    )

    prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    results.append({
        "id": tc["id"],
        "input": input_text,
        "prediction_lora": prediction
    })

with open("lora_results.json", "w") as f:
    json.dump(results, f, indent=4)
