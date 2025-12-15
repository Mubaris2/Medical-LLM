from huggingface_hub import hf_hub_download
import os

print("Downloading LoRA weights...")
path = hf_hub_download(
    repo_id="Mubaris2004/Medical-LLM-LoRA",
    filename="adapter_model.safetensors",
)

os.makedirs("final_lora_weights", exist_ok=True)
target = "final_lora_weights/adapter_model.safetensors"

if not os.path.exists(target):
    os.rename(path, target)

print("LoRA weights downloaded successfully!")
