#!/usr/bin/env python
"""
code/Training/train_fever_rag_batched.py

Batched RAG-style training on FEVER v1.0 (train split), using a fixed FAISS
index of 10k pages. Saves its final model to:

    code/Training/batched_checkpoints/fever_rag_batched.pt
"""

import json, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset, Dataset

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from verifier.verifier import RAGVerifier
from retriever.retriever import Retriever

FAISS_PATH = SRC_DIR.parent / "Embeddings/fever_pages.faiss"
IDS_JSON   = SRC_DIR.parent / "Embeddings/fever_pages_ids.json"
DPR_CONFIG = "psgs_w100.nq.exact"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16        # adjust for your GPU
EPOCHS     = 2
TOP_K      = 1
LR         = 5e-5

LABEL_TOK = {
    "SUPPORTS":         "<supports>",
    "REFUTES":          "<refutes>",
    "NOT ENOUGH INFO":  "<unknown>",
}


def collate_fever(batch):
    """
    Convert a list of FEVER examples to batch dict:
      - 'claims': list[str]
      - 'labels': list[str]  (token strings)
    """
    return {
        "claims": [ex["claim"] for ex in batch],
        "labels": [LABEL_TOK[ex["label"]] for ex in batch],
    }


def main():
    print(f"Using device: {DEVICE}\n")

    fever_ds = load_dataset("fever", "v1.0", split="train")
    loader = DataLoader(
        fever_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fever,
        drop_last=True,
    )
    print(f"Loaded {len(fever_ds):,} FEVER examples, batching {BATCH_SIZE}.\n")

    # ────────────────────────────────────────────────────────────────────
    # 2) Build 10k-passage DPR corpus subset
    # ────────────────────────────────────────────────────────────────────
    with open(IDS_JSON) as f:
        keep_ids = set(json.load(f))

    stream = load_dataset("wiki_dpr", DPR_CONFIG, split="train", streaming=True)
    texts = []
    for item in tqdm(stream, total=len(keep_ids), desc="Collecting DPR"):
        if item["id"] in keep_ids:
            texts.append(item["text"])
            if len(texts) == len(keep_ids):
                break

    corpus_ds = Dataset.from_dict({"text": texts})
    print(f"Built DPR corpus subset with {len(corpus_ds):,} passages.\n")

    # ────────────────────────────────────────────────────────────────────
    # 3) Instantiate models & optimizer
    # ────────────────────────────────────────────────────────────────────
    verifier  = RAGVerifier(device=DEVICE)
    retriever = Retriever(str(FAISS_PATH), corpus=corpus_ds, device=DEVICE)

    optimizer = AdamW(
        list(retriever.q.parameters()) + list(verifier.parameters()),
        lr=LR
    )
    verifier.train()
    retriever.q.train()

    # ────────────────────────────────────────────────────────────────────
    # 4) Batched training loop
    # ────────────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0

        for step, batch in enumerate(tqdm(loader, desc="Training")):
            claims, labels = batch["claims"], batch["labels"]

            # a) Retrieve top-k passages for the batch
            docs, _ = retriever(claims, k=TOP_K)
            contexts = [
                " ".join(d["text"] for d in doc_list)
                for doc_list in docs
            ]

            # b) Tokenize all claim+context pairs at once
            input_texts = [
                f"{c} {verifier.tokenizer.eos_token} {ctx}"
                for c, ctx in zip(claims, contexts)
            ]
            inputs = verifier.tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(DEVICE)

            # c) Convert label tokens to IDs
            label_ids = verifier.tokenizer(
                labels,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )["input_ids"].to(DEVICE)

            # d) Forward & compute loss
            out = verifier.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=label_ids,
            )
            loss = out.loss

            # e) Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(retriever.q.parameters()) + list(verifier.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % 100 == 0:
                avg = running_loss / 100
                print(f" step {step+1:<6}  avg_loss={avg:.4f}")
                running_loss = 0.0

    # ────────────────────────────────────────────────────────────────────
    # 5) Save checkpoint to new dir & filename
    # ────────────────────────────────────────────────────────────────────
    ckpt_dir = SRC_DIR / "Training" / "batched_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "fever_rag_batched.pt"

    torch.save({
        "verifier_state": verifier.state_dict(),
        "query_encoder_state": retriever.q.state_dict(),
    }, ckpt_file)

    print(f"\nCheckpoint saved to {ckpt_file}")

if __name__ == "__main__":
    main()
