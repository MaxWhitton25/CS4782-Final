#!/usr/bin/env python

#Embedding_Verification.py

#Build a FAISS GPU index over the ~5K Wikipedia pages used in FEVER,
#with fast local filtering and batched embedding.

import os
import json
import time
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

def embed_passages(texts, tokenizer, model, device):
    """
    Return CLS embeddings for a list of texts as a NumPy array.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].cpu().numpy()

def move_index_to_gpu(cpu_index):
    """
    Move a FAISS index to GPU if possible.
    """
    if hasattr(faiss, "StandardGpuResources"):
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        return gpu_index
    else:
        return cpu_index

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for BERT: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print()

    print("Extracting FEVER page titles…")
    fever_pages = extract_fever_pages()
    print(f" → {len(fever_pages)} unique FEVER pages.\n")

    print("Loading top 100K Wikipedia passages locally (no streaming)…")
    wiki_data = load_dataset(
        "wiki_dpr",
        "psgs_w100.nq.exact",
        split="train[:100000]",
        trust_remote_code=True,
    )

    print("Filtering passages that match FEVER titles…")
    matched = [ex for ex in wiki_data if ex["title"] in fever_pages]
    print(f" → Matched {len(matched)} passages.\n")

    print("Loading BERT embedder…")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

    print("Embedding passages in batches…")
    BATCH_SIZE = 32
    embeddings = []
    ids = []
    start_time = time.time()

    for i in tqdm(range(0, len(matched), BATCH_SIZE), desc="Embedding"):
        batch = matched[i:i + BATCH_SIZE]
        texts = [ex["text"] for ex in batch]
        batch_ids = [ex["id"] for ex in batch]
        batch_embs = embed_passages(texts, tokenizer, model, device)
        embeddings.append(batch_embs)
        ids.extend(batch_ids)

    print(f"\n → Total embedded passages: {len(ids)} in {int(time.time() - start_time)}s\n")

    matrix = np.vstack(embeddings).astype('float32')
    faiss.normalize_L2(matrix)

    d = matrix.shape[1]
    print(f"Building FAISS index with vector dimension {d}…")
    cpu_index = faiss.IndexFlatIP(d)
    index = move_index_to_gpu(cpu_index)
    index.add(matrix)
    print(f"Built FAISS index with {index.ntotal} vectors.\n")

    print("Saving FAISS index and metadata…")
    cpu_final_index = faiss.index_gpu_to_cpu(index)

    idx_path = "fever_pages.index"
    meta_path = "fever_pages_ids.json"
    faiss.write_index(cpu_final_index, idx_path)

    with open(meta_path, "w") as f:
        json.dump(ids, f)

    print("Done generating FAISS index of FEVER pages.")

if __name__ == "__main__":
    main()
