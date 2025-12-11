from unsloth import FastLanguageModel
import torch
import json

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length = 1024,
    dtype = torch.float16,
    load_in_4bit = True,
)

with open("../DataEngineering/eval_testcases.json") as f:
    testcases = json.load(f)

results = []

for tc in testcases:
    input_text = tc["input"]

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens = 150,
        temperature = 0.2,
    )

    prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    results.append({
        "id": tc["id"],
        "input": input_text,
        "prediction_base": prediction
    })

with open("base_results.json", "w") as f:
    json.dump(results, f, indent=4)
