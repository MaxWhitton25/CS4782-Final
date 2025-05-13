#!/usr/bin/env python
"""
code/Training/train_fever_rag_batched.py

Batched RAG-style training on FEVER v1.0 (train split with 80/20 train-test split),
using a fixed FAISS index of 10k pages. Saves its final model to:

    code/Training/batched_checkpoints/fever_rag_batched.pt
"""
import json, sys, os
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset, Dataset

# Set a fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

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
TRAIN_RATIO = 0.8     # 80% train, 20% test split

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


def create_train_test_split(dataset, train_ratio=0.8, seed=42):
    """
    Create train and test splits from a dataset.
    
    Args:
        dataset: The original dataset
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, test_dataset
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # Shuffle indices
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    # Calculate split point
    split = int(np.floor(train_ratio * dataset_size))
    
    # Create train and test indices
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset


def main():
    print(f"Using device: {DEVICE}\n")

    # Load the FEVER dataset (train split)
    full_dataset = load_dataset("fever", "v1.0", split="train")
    
    # Create train/test split (80/20)
    train_dataset, test_dataset = create_train_test_split(
        full_dataset, 
        train_ratio=TRAIN_RATIO, 
        seed=RANDOM_SEED
    )
    
    print(f"Original dataset size: {len(full_dataset):,}")
    print(f"Train split size: {len(train_dataset):,}")
    print(f"Test split size: {len(test_dataset):,}")
    
    # Save the test indices for reproducibility
    ckpt_dir = SRC_DIR / "Training" / "batched_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract and save test indices
    test_indices = test_dataset.indices
    with open(ckpt_dir / "fever_rag_split_test_indices.json", "w") as f:
        json.dump(test_indices, f)
    
    # Create dataloader for the training subset
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fever,
        drop_last=True,
    )
    print(f"Created dataloader with batch size {BATCH_SIZE}.\n")

    # 2) Build 10k-passage DPR corpus subse
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

    # 3) Instantiate models & optimizer
    verifier  = RAGVerifier(device=DEVICE)
    retriever = Retriever(str(FAISS_PATH), corpus=corpus_ds, device=DEVICE)

    optimizer = AdamW(
        list(retriever.q.parameters()) + list(verifier.parameters()),
        lr=LR
    )
    verifier.train()
    retriever.q.train()

    # 4) Batched training loop
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

    # 5) Save checkpoint to new dir & filename
    ckpt_file = ckpt_dir / "fever_rag_split.pt"

    torch.save({
        "verifier_state": verifier.state_dict(),
        "query_encoder_state": retriever.q.state_dict(),
        "random_seed": RANDOM_SEED,
        "train_test_ratio": TRAIN_RATIO
    }, ckpt_file)

    print(f"\nCheckpoint saved to {ckpt_file}")
    print(f"Test indices saved to {ckpt_dir / 'fever_rag_split_test_indices.json'}")

if __name__ == "__main__":
    main()
