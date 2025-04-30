#!/usr/bin/env python
"""
src/Training/train_fever_rag.py
────────────────────────────────
End-to-end RAG-style training on FEVER v1.0 (train split) with

  • fixed 10 k-page FAISS index (Embeddings/fever_pages.index)
  • DPR passage subset streamed on-the-fly
  • joint optimisation of BertQueryEncoder + BART verifier
  • no retrieval supervision (only label loss)
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset, Dataset

# ────────────────────────────────────────────────────────────────────
#  import path so "from retriever.retriever import Retriever" works
# ────────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from verifier.verifier import RAGVerifier
from retriever.retriever import Retriever

# ----------------------------- CONFIG ------------------------------
FAISS_PATH   = SRC_DIR.parent / "Embeddings/fever_pages.faiss"
IDS_JSON     = SRC_DIR.parent / "Embeddings/fever_pages_ids.json"
DPR_CONFIG   = "psgs_w100.nq.exact"      # same config used for embeddings
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE   = 8
EPOCHS       = 2
TOP_K        = 1
LR           = 5e-5

LABEL_TOK = {
    "SUPPORTS": "<supports>",
    "REFUTES": "<refutes>",
    "NOT ENOUGH INFO": "<unknown>",
    0: "<supports>",
    1: "<refutes>",
    2: "<unknown>",
}
# -------------------------------------------------------------------


# ╭─────────────────────────────────────────────────────────────────╮
# │ 1.  FEVER train split (no filtering)                           │
# ╰─────────────────────────────────────────────────────────────────╯
print("Loading FEVER v1.0 train split …")
fever_ds = load_dataset("fever", "v1.0", split="train")
print(f" → {len(fever_ds):,} claims loaded.")

def collate(batch):
    claims = [ex["claim"] for ex in batch]
    labels = [LABEL_TOK[ex["label"]] for ex in batch]
    return {"claims": claims, "labels": labels}

loader = DataLoader(
    fever_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
)


# ╭─────────────────────────────────────────────────────────────────╮
# │ 2.  Build tiny DPR corpus (10 k passages)                      │
# ╰─────────────────────────────────────────────────────────────────╯
print("\nBuilding DPR corpus subset used by FAISS index …")
with open(IDS_JSON) as f:
    keep_ids = set(json.load(f))               # 10 k passage IDs

stream = load_dataset("wiki_dpr", DPR_CONFIG, split="train", streaming=True)

texts = []
for item in tqdm(stream, total=len(keep_ids)):
    if item["id"] in keep_ids:
        texts.append(item["text"])
        if len(texts) == len(keep_ids):
            break

print(f" → collected {len(texts):,} passages.\n")
corpus_ds = Dataset.from_dict({"text": texts})   # single-column dataset


# ╭─────────────────────────────────────────────────────────────────╮
# │ 3.  Models & optimiser                                         │
# ╰─────────────────────────────────────────────────────────────────╯
verifier  = RAGVerifier(device=DEVICE)
retriever = Retriever(str(FAISS_PATH), corpus=corpus_ds, device=DEVICE)

params = list(retriever.q.parameters()) + list(verifier.parameters())
optim  = AdamW(params, lr=LR)

retriever.q.train()
verifier.train()

# ╭─────────────────────────────────────────────────────────────────╮
# │ 4.  Training loop                                              │
# ╰─────────────────────────────────────────────────────────────────╯
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    running = 0.0
    for step, batch in enumerate(tqdm(loader)):
        claims, golds = batch["claims"], batch["labels"]

        # —— retrieve top-k passages
        docs, _ = retriever(claims, k=TOP_K)
        contexts = [
            " ".join(doc["text"] for doc in doc_list) if doc_list else ""
            for doc_list in docs
        ]

        # —— compute loss (per example for clarity)
        losses = [
            verifier.train_run(claim, ctx, lbl, DEVICE)
            for claim, ctx, lbl in zip(claims, contexts, golds)
        ]
        loss = torch.stack(losses).mean()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()

        running += loss.item()
        if (step + 1) % 200 == 0:
            print(f"  step {step+1:<6}  avg_loss={running/200:.4f}")
            running = 0.0


# ╭─────────────────────────────────────────────────────────────────╮
# │ 5.  Save checkpoint                                            │
# ╰─────────────────────────────────────────────────────────────────╯
ckpt_dir = SRC_DIR / "Training" / "checkpoints"
ckpt_dir.mkdir(exist_ok=True)
ckpt_file = ckpt_dir / "fever_rag.pt"

torch.save(
    {
        "verifier_state": verifier.state_dict(),
        "query_encoder_state": retriever.q.state_dict(),
    },
    ckpt_file,
)

print(f"\nCheckpoint saved to {ckpt_file}")
