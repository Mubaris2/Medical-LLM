import streamlit as st
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

st.title("📊 Medical LLM - Model Fine-Tuning Evaluation Dashboard")

st.write("""
This page visualizes evaluation metrics comparing:
- **Base Model**
- **LoRA Fine-tuned Model**
- **LoRA + RAG Model**

Data is loaded directly from `Evaluation/evaluation_results.json`.
""")

BASE_DIR = Path(__file__).resolve().parent[1]
FILE_PATH = BASE_DIR / "data" / "evaluation_results.json"

try:
    with open(FILE_PATH, "r") as f:
        results = json.load(f)
except Exception as e:
    st.error(f"❌ Could not load {FILE_PATH}\nError: {e}")
    st.stop()

df = pd.DataFrame(results)

st.subheader("📄 Raw Evaluation Data")
st.dataframe(df, use_container_width=True)

metric_names = list(df["metrics"][0].keys())

def extract_metric(metric_name):
    base_vals = []
    lora_vals = []
    rag_vals = []

    for item in df["metrics"]:
        metric_block = item.get(metric_name, {})
        if isinstance(metric_block, dict):
            base_vals.append(metric_block.get("base", 0))
            lora_vals.append(metric_block.get("lora", 0))
            rag_vals.append(metric_block.get("rag", 0))
        else:
            base_vals.append(0)
            lora_vals.append(0)
            rag_vals.append(metric_block)

    return np.array(base_vals), np.array(lora_vals), np.array(rag_vals)

st.subheader("📈 Metric Comparison Graphs")

for metric in metric_names:
    st.markdown(f"### 🔹 {metric.replace('_', ' ').title()}")

    base_vals, lora_vals, rag_vals = extract_metric(metric)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(base_vals))

    ax.plot(x, base_vals, label="Base", marker="o")
    ax.plot(x, lora_vals, label="LoRA", marker="o")
    ax.plot(x, rag_vals, label="RAG", marker="o")

    ax.set_xlabel("Test Case Index")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

st.markdown("---")

st.subheader("📊 Average Scores Across All Test Cases")

avg_summary = {
    "metric": [],
    "base": [],
    "lora": [],
    "rag": []
}

for metric in metric_names:
    base_vals, lora_vals, rag_vals = extract_metric(metric)

    avg_summary["metric"].append(metric)
    avg_summary["base"].append(round(base_vals.mean(), 4))
    avg_summary["lora"].append(round(lora_vals.mean(), 4))
    avg_summary["rag"].append(round(rag_vals.mean(), 4))

summary_df = pd.DataFrame(avg_summary)
st.dataframe(summary_df, use_container_width=True)

st.subheader("🏆 Overall Average Comparison")

fig, ax = plt.subplots(figsize=(8, 5))

bar_width = 0.25
indices = np.arange(len(summary_df))

ax.bar(indices, summary_df["base"], width=bar_width, label="Base")
ax.bar(indices + bar_width, summary_df["lora"], width=bar_width, label="LoRA")
ax.bar(indices + 2*bar_width, summary_df["rag"], width=bar_width, label="RAG")

ax.set_xticks(indices + bar_width)
ax.set_xticklabels(summary_df["metric"], rotation=45, ha="right")
ax.set_ylabel("Score")
ax.legend()

st.pyplot(fig)

st.success("Evaluation dashboard loaded successfully!")