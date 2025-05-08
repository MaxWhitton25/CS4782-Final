#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
import numpy as np
from datasets import Dataset

# allow import of QueryEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/retriever"))
from QueryEncoder import BertQueryEncoder


class Retriever(nn.Module):
    """
    Dense retriever using FAISS, with GPU compatibility.

    Args:
        vd_path (str): path to vector database (.faiss or .pt embedding file)
        corpus (Dataset): HF Dataset of passages, with 'text' field
        device (str or torch.device): device for query encoder and embeddings
    """

    def __init__(self, vd_path: str, corpus: Dataset, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        # Load FAISS index or embeddings
        if vd_path.endswith((".faiss", ".index")):
            # prebuilt FAISS index on CPU
            print(f"Loading FAISS index from {vd_path}")
            self.index = faiss.read_index(vd_path)
            print(f"Loaded index: ntotal={self.index.ntotal}, dim={self.index.d}")
            self.embeddings = None
        elif vd_path.endswith(".pt"):
            # load precomputed embeddings and build index
            print(f"Loading embeddings from {vd_path}")
            data = torch.load(vd_path, map_location="cpu")
            emb = data["embeddings"]  # CPU tensor
            # move tensor to device for cosine updates
            self.embeddings = emb.to(self.device)
            print(f"Converting embeddings to FAISS (dim={emb.shape[1]})")
            emb_np = emb.numpy().astype("float32")
            self.index = faiss.IndexFlatIP(emb_np.shape[1])
            self.index.add(emb_np)
        else:
            raise ValueError(f"Unsupported vector db type: {vd_path}")

        self.corpus = corpus
        # initialize query encoder on device
        self.q = BertQueryEncoder(device=self.device)

    def forward(self, queries, k: int = 1):
        """
        Retrieve top-k passages for a batch of input queries.

        Args:
            queries (List[str]): batch of query strings
            k (int): number of top passages to return

        Returns:
            docs: List[List[record]]  // top-k passages per query
            probs: torch.Tensor       // (batch, k) retrieval probabilities
        """
        # 1) encode queries to embeddings on device
        q_emb = self.q(queries).to(self.device)  # (batch, dim)
        q_emb = F.normalize(q_emb, p=2, dim=-1)

        # 2) convert to CPU numpy for FAISS
        q_np = q_emb.detach().cpu().numpy().astype("float32")
        D, I = self.index.search(q_np, k)  # D: (batch,k), I: indices

        # 3) convert sims back to torch on device
        sims = F.normalize(torch.from_numpy(D).to(self.device), p=2, dim=-1)

        docs = [[self.corpus[int(idx)] for idx in row] for row in I]

        # Mihir commented this out because it's redundant
        # if hasattr(self, 'embeddings'):
        #     I_pt = torch.from_numpy(I).to(torch.long)
        #     emb_vectors = self.embeddings[I_pt]       # (batch,k,dim)
        #     q_exp = q_emb.unsqueeze(1)
        #     sims =  torch.sum(q_exp * emb_vectors, axis = -1)

        # 6. probabilities
        probs = F.softmax(sims, dim=-1)
        return docs, probs.to(self.device)
