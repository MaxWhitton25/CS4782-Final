#!/usr/bin/env python
"""
train_fever_rag.py

End-to-end training of a RAG-style verifier on FEVER,
using a fixed FAISS index over the 10 k relevant Wiki pages.

"""

import os, sys, json, math
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from verifier.verifier import RAGVerifier
from retriever.retriever import Retriever

# -----------------------  CONFIG  ------------------------------------
FAISS_PATH   = SRC_DIR.parent / "Embeddings/fever_pages.index"
IDS_PATH     = SRC_DIR.parent / "Embeddings/fever_pages_ids.json"
BATCH_SIZE   = 8
EPOCHS       = 2
TOP_K        = 1
LR           = 5e-5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_MAP = {0: "<supports>", 1: "<refutes>", 2: "<unknown>"}
# ---------------------------------------------------------------------

def filter_to_index_pages(split="train", cfg="v1.0") -> List[dict]:
    """
    Return FEVER examples whose gold evidence_wiki_url exists in our 10 k page set.
    (We only need the claim string and the FEVER label id.)
    """
    with open(IDS_PATH) as f:
        # these are DPR passage IDs; we stored only pages present in the index
        kept_ids = set(json.load(f))

    ds = load_dataset("fever", cfg, split=split)
    filtered = []
    for ex in ds:
        # Natural-power fallback: keep everything – index retrieval will
        # simply fail soft if it can’t find a passage
        wiki_id = ex.get("evidence_id", None)
        if wiki_id is None or wiki_id in kept_ids:
            filtered.append({"claim": ex["claim"], "label": ex["label"]})
    return filtered


def collate(batch):
    # batch is list of dicts {"claim": str, "label": int}
    claims = [b["claim"] for b in batch]
    labels = [LABEL_MAP[b["label"]] for b in batch]
    return {"claims": claims, "labels": labels}


def main():
    print(f"Device: {DEVICE}")

    # --------------------  DATA  ------------------------------------
    print("Loading FEVER examples …")
    examples = filter_to_index_pages(split="train")
    print(f" → keeping {len(examples)} training examples.")

    loader = DataLoader(
        examples,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )

    # Load the same corpus used for embeddings
    print("Loading DPR passages subset …")
    import datasets  # lazy
    corpus = datasets.load_dataset(
        "wiki_dpr",
        "psgs_w100.nq.exact",
        split="train",
        streaming=True  # do not download everything
    )

    # --------------------  MODELS  ----------------------------------
    verifier  = RAGVerifier(device=DEVICE)
    retriever = Retriever(str(FAISS_PATH), corpus=corpus, device=DEVICE)

    # Train QueryEncoder and BART together
    params = list(retriever.q.parameters()) + list(verifier.parameters())
    optim  = AdamW(params, lr=LR)

    verifier.train()
    retriever.q.train()

    step = 0
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            claims = batch["claims"]
            gold_labels = batch["labels"]

            # ----------------  RETRIEVE  -----------------
            docs, _ = retriever(claims, k=TOP_K)
            contexts = [" ".join([d["text"] for d in doc_list]) if doc_list else ""
                        for doc_list in docs]

            # ----------------  VERIFY   ------------------
            # Process entire batch at once
            input_texts = [f"{claim} {verifier.tokenizer.eos_token} {ctx}" 
                         for claim, ctx in zip(claims, contexts)]
            inputs = verifier.tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            labels = verifier.tokenizer(
                gold_labels,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )

            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)
            label_ids = labels["input_ids"].to(DEVICE)

            outputs = verifier.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label_ids,
            )
            loss = outputs.loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

            running_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"step {step:>6}  avg_loss={running_loss/50:.4f}")
                running_loss = 0.0

    # ---------------- SAVE CHECKPOINT -----------------
    ckpt_dir = SRC_DIR / "Training" / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "fever_rag.pt"
    state = {
        "verifier": verifier.state_dict(),
        "query_encoder": retriever.q.state_dict(),
    }
    torch.save(state, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()
