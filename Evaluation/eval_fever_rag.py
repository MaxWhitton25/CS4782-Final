#!/usr/bin/env python
"""
Evaluation/eval_fever_rag.py
────────────────────────────
Accuracy & P/R/F1 for a trained RAG verifier on FEVER v1.0.

▸ Metrics for two systems
    1. RAG  = claim + top-k retrieved passage(s)
    2. BASE = claim only (blank context)

▸ Outputs
    • Console metrics
    • CSV with one row per example: claim, gold, rag_pred, base_pred

Run:
    python Evaluation/eval_fever_rag.py
Optional flags:
    --batch_size 16         # GPU memory permitting
    --device cuda           # or cpu
    --csv_out preds.csv     # custom name
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
#  Project paths  (repo_root/
#      ├─ src/
#      └─ Evaluation/eval_fever_rag.py  ← this file)
# ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from verifier.verifier import RAGVerifier
from retriever.retriever import Retriever

# --------------------------- CONSTANTS ------------------------
FAISS_PATH = REPO_ROOT / "Embeddings/fever_pages.faiss"
IDS_JSON   = REPO_ROOT / "Embeddings/fever_pages_ids.json"
DPR_CONFIG = "psgs_w100.nq.exact"          # same as training

LABEL_TOK = {
    "SUPPORTS": "<supports>",
    "REFUTES": "<refutes>",
    "NOT ENOUGH INFO": "<unknown>",
    0: "<supports>",
    1: "<refutes>",
    2: "<unknown>",
}
INV_LABEL = {
    "<supports>": "SUPPORTS",
    "<refutes>": "REFUTES",
    "<unknown>": "NOT ENOUGH INFO",
}
ALLOWED_TOKENS = set(INV_LABEL.keys())


# ──────────────────────────────────────────────────────────────
#  Utility: build tiny DPR corpus used by FAISS index
# ──────────────────────────────────────────────────────────────
def build_corpus() -> Dataset:
    with open(IDS_JSON) as f:
        keep_ids = set(json.load(f))          # 10 000 ids
    stream = load_dataset("wiki_dpr", DPR_CONFIG, split="train", streaming=True)

    texts: List[str] = []
    for item in tqdm(stream, total=len(keep_ids), desc="Collect DPR passages"):
        if item["id"] in keep_ids:
            texts.append(item["text"])
            if len(texts) == len(keep_ids):
                break
    return Dataset.from_dict({"text": texts})


# ──────────────────────────────────────────────────────────────
#  Batched classification helper
# ──────────────────────────────────────────────────────────────
def classify_batch(
    verifier: RAGVerifier,
    claims: List[str],
    contexts: List[str],
    device: str,
) -> List[str]:
    """
    Return one of the three canonical special tokens per example.
    """
    tok = verifier.tokenizer
    eos = tok.eos_token
    enc = tok(
        [f"{c} {eos} {ctx}" for c, ctx in zip(claims, contexts)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        # let BART autoregress until it hits EOS
        outs = verifier.model.generate(
            **enc,
            max_length=4,        # 1-2 sub-tokens + </s>
            num_beams=1,
            early_stopping=True,
        )

    decoded = tok.batch_decode(outs, skip_special_tokens=False)

    def to_label(txt: str) -> str:
        txt = txt.strip()  # drop leading space
        for cand in ALLOWED_TOKENS:
            if cand in txt:
                return cand
        return "<unknown>"   # fall-back

    return [to_label(t) for t in decoded]


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv_out", default="preds_fever_val.csv")
    args = parser.parse_args()

    device = args.device
    bs     = args.batch_size
    csv_out = REPO_ROOT / "Evaluation" / args.csv_out

    # 1) FEVER split -------------------------------------------------
    try:
        fever_ds = load_dataset("fever", "v1.0", split="validation")
    except Exception:
        print("⚠️  validation split missing – taking first 5 000 examples from train.")
        fever_ds = load_dataset("fever", "v1.0", split="train[:5000]")
    print(f"🗂  Loaded {len(fever_ds):,} examples")

    # 2) Retriever corpus & FAISS ------------------------------------
    print("🔧 Preparing retriever …")
    corpus_ds = build_corpus()
    retriever = Retriever(str(FAISS_PATH), corpus=corpus_ds, device=device)

    # 3) Verifier & checkpoint ---------------------------------------
    verifier = RAGVerifier(device=device)
    ckpt = torch.load(REPO_ROOT / "src/Training/checkpoints/fever_rag.pt", map_location=device)
    verifier.load_state_dict(ckpt["verifier_state"])
    retriever.q.load_state_dict(ckpt["query_encoder_state"])

    verifier.eval()
    retriever.q.eval()

    # 4) Loop ---------------------------------------------------------
    y_true, y_pred_rag, y_pred_base = [], [], []

    for i in tqdm(range(0, len(fever_ds), bs), desc="Evaluating"):
        batch   = fever_ds[i : i + bs]
        claims  = batch["claim"]
        gold_in = batch["label"]          # ints 0/1/2
        gold    = [LABEL_TOK[g] for g in gold_in]

        # RAG retrieval
        docs, _ = retriever(claims, k=1)
        contexts = [
            " ".join(d["text"] for d in doc_list) if doc_list else "" for doc_list in docs
        ]

        # Predictions
        rag_pred  = classify_batch(verifier, claims, contexts, device)
        base_pred = classify_batch(verifier, claims, [""] * len(claims), device)

        y_true.extend(gold)
        y_pred_rag.extend(rag_pred)
        y_pred_base.extend(base_pred)

    # 5) Metrics ------------------------------------------------------
    def report(name, preds):
        print(f"\n── {name} ───────────────────────────")
        acc = accuracy_score(y_true, preds)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, preds, labels=list(ALLOWED_TOKENS), average="macro", zero_division=0
        )
        print(f"accuracy        : {acc:.4f}")
        print(f"macro-precision : {p:.4f}")
        print(f"macro-recall    : {r:.4f}")
        print(f"macro-F1        : {f:.4f}")
        print("\nDetailed report:")
        print(
            classification_report(
                y_true,
                preds,
                labels=list(ALLOWED_TOKENS),
                target_names=[INV_LABEL[t] for t in ALLOWED_TOKENS],
                digits=3,
                zero_division=0,
            )
        )

    report("RAG (k=1)", y_pred_rag)
    report("Baseline (no context)", y_pred_base)

    # 6) CSV ----------------------------------------------------------
    header = ["id", "claim", "gold", "rag_pred", "base_pred"]
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, (ex, g, rp, bp) in enumerate(zip(fever_ds, y_true, y_pred_rag, y_pred_base)):
            writer.writerow([idx, ex["claim"], INV_LABEL[g], INV_LABEL[rp], INV_LABEL[bp]])
    print(f"📄 Predictions written to {csv_out.resolve()}")


if __name__ == "__main__":
    main()
