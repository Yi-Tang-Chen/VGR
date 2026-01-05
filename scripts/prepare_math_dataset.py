import os, json
from datasets import load_dataset
from tqdm import tqdm

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
