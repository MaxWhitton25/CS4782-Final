#!/usr/bin/env python
"""
Embedding_Verification.py

Build a FAISS index over the ~5K Wikipedia pages used in FEVER,
*streaming* wiki_dpr and early-stopping as soon as we've covered each page once.
"""

import os
import json
import numpy as np
import faiss
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer, BertModel

def extract_fever_pages(split="train", config="v1.0"):
    """
    Load FEVER and collect all unique page titles from evidence_wiki_url.
    """
    ds = load_dataset("fever", config, split=split, trust_remote_code=True)
    pages = set()
    for ex in ds:
        url = ex.get("evidence_wiki_url", "")
        if url:
            title = url.rsplit("/", 1)[-1]
            pages.add(title)
    return pages

def embed_passage(text, tokenizer, model, device):
    """
    Return the CLS embedding for a piece of text.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].cpu().numpy().squeeze(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Extracting FEVER page titles…")
    fever_pages = extract_fever_pages()
    print(f" → {len(fever_pages)} unique pages found.\n")

    print("Streaming wiki_dpr (psgs_w100.nq.exact)…")
    stream = load_dataset(
        "wiki_dpr", 
        "psgs_w100.nq.exact", 
        split="train", 
        streaming=True,
        trust_remote_code=True,
    )
    remaining = set(fever_pages)     # pages we still need
    embeddings = []
    ids = []

    print("Loading BERT embedder…")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

    print("Embedding passages (one per FEVER page)…")
    for item in tqdm(stream):
        title = item["title"]
        if title in remaining:
            emb = embed_passage(item["text"], tokenizer, model, device)
            embeddings.append(emb)
            ids.append(item["id"])
            remaining.remove(title)
            # once we've covered every page, break
            if not remaining:
                break

    print(f"\n → Embedded {len(ids)} passages, covering all pages.\n")

    matrix = np.vstack(embeddings)
    faiss.normalize_L2(matrix)
    d = matrix.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(matrix)
    print(f"Built FAISS index with {index.ntotal} vectors.\n")

    idx_path = "fever_pages.index"
    meta_path = "fever_pages_ids.json"
    print(f"Saving index to {idx_path} and IDs to {meta_path}…")
    faiss.write_index(index, idx_path)
    with open(meta_path, "w") as f:
        json.dump(ids, f)

    print("\nDone! You now have a tiny index (~5K vectors) instead of 50M+.")

if __name__ == "__main__":
    main()
