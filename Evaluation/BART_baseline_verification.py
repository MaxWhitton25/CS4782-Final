import sys
import os

# allow imports from code/ (one level up)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# now import from code/verifier/verifier.py
from verifier.verifier import RAGVerifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
SUBSET_SIZE = 1000   # small eval set

# map FEVER labels to your special tokens
LABEL_MAP = {
    0: "<supports>",
    1: "<refutes>",
    2: "<unknown>",
}

def collate_fn(batch):
    claims = [ex["claim"] for ex in batch]
    gold_labels = [LABEL_MAP[ex["label"]] for ex in batch]
    return {"claims": claims, "gold_labels": gold_labels}

def main():
    print(f"Using device: {DEVICE}\n")

    print("Loading FEVER test split…")
    ds = load_dataset("fever", "v1.0", split="test")
    ds = ds.shuffle(seed=42).select(range(SUBSET_SIZE))
    print(f" → selected {len(ds)} examples for quick eval\n")

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("Initializing RAGVerifier…")
    verifier = RAGVerifier(model_name="facebook/bart-base", device=DEVICE)
    verifier.model.eval()

    # ZERO-CONTEXT BASELINE
    records = []
    correct = 0

    print("Running zero-context classification…")
    for batch in tqdm(loader, total=len(loader)):
        for claim, gold in zip(batch["claims"], batch["gold_labels"]):
            pred = verifier.classify(claim, context="")
            is_correct = int(pred == gold)
            correct += is_correct
            records.append({
                "claim": claim,
                "gold_label": gold,
                "pred_label": pred,
                "correct": is_correct,
            })

    total = len(records)
    accuracy = correct / total
    print(f"\nZero-context accuracy over {total} examples: {accuracy:.4%}\n")

    out_csv = os.path.join(os.path.dirname(__file__), "fever_zero_context_baseline.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Per-example results saved to {out_csv}")

if __name__ == "__main__":
    main()
