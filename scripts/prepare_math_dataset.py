import os, json
from datasets import load_dataset
from tqdm import tqdm

import os, json
from datasets import load_dataset

os.makedirs("datasets/gsm8k", exist_ok=True)
ds = load_dataset("openai/gsm8k", "main")  # GSM8K :contentReference[oaicite:4]{index=4}
for split in ["train", "test"]:
    out = f"datasets/gsm8k/{split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds[split]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print("Saved GSM8K to datasets/gsm8k/{train,test}.jsonl")

# EleutherAI 的 MATH 鏡像（可用，且有 train/test）
DATASET_NAME = "EleutherAI/hendrycks_math"
SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

OUT_ROOT = "datasets/MATH"

def dump_split(subject: str, split: str):
    ds = load_dataset(DATASET_NAME, subject, split=split)
    out_dir = os.path.join(OUT_ROOT, split, subject)
    os.makedirs(out_dir, exist_ok=True)

    for i, ex in enumerate(tqdm(ds, desc=f"{subject}/{split}")):
        # ex: {problem, solution, level, type}
        out_path = os.path.join(out_dir, f"{i:05d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ex, f, ensure_ascii=False)

def main():
    for subject in SUBJECTS:
        dump_split(subject, "train")
        dump_split(subject, "test")

    # quick sanity check
    total = 0
    for root, _, files in os.walk(OUT_ROOT):
        total += sum(1 for fn in files if fn.endswith(".json"))
    print(f"[OK] Wrote {total} json files under {OUT_ROOT}/")

if __name__ == "__main__":
    main()
