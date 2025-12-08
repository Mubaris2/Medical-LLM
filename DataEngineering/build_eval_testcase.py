import json
import random

def extract_testcase(example):
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()
    inst = example.get("instruction", "").strip()

    return {
        "id": None,
        "source": "SFT_VAL",
        "instruction": inst,
        "input": inp,
        "expected_keywords": out
    }

with open("sft_val.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cases = [extract_testcase(row) for row in data]
random.shuffle(cases)
cases = cases[:100]
for i, c in enumerate(cases, start=1):
    c["id"] = i

with open("eval_testcases.json", "w", encoding="utf-8") as f:
    json.dump(cases, f, indent=4, ensure_ascii=False)