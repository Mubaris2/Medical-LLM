from unsloth import FastLanguageModel
import torch

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
lora_path = "final_lora_weights"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model,
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
)

model.load_adapter(lora_path)
model.save_pretrained("merged_model", tokenizer)