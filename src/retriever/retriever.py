import os, sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
import numpy as np
from datasets import Dataset

# allow "from QueryEncoder import BertQueryEncoder"
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/retriever"))
from QueryEncoder import BertQueryEncoder


class Retriever(nn.Module):
    """
    Simple dense retriever:
      • encodes a batch of queries with BertQueryEncoder
      • searches a FAISS index (pre-built or converted from .pt)
      • returns (docs, similarity_scores)
    """

    def __init__(self, vd_path: str, corpus: Dataset, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        # ── Load vector database ─────────────────────────────────────
        if vd_path.endswith((".faiss", ".index")):
            print("Loading FAISS index …")
            self.index = faiss.read_index(vd_path)
            print("  loaded index with", self.index.ntotal, "vectors.")
        elif vd_path.endswith(".pt"):
            data = torch.load(vd_path, map_location="cpu")
            self.embeddings = data["embeddings"]          # torch tensor
            print("Converting .pt embeddings to FAISS …")
            emb_np = self.embeddings.numpy().astype("float32")
            idx    = faiss.IndexFlatL2(emb_np.shape[1])
            idx.add(emb_np)
            self.index = idx
        else:
            raise ValueError(f"Unsupported file type: {vd_path}")

        # corpus is a HuggingFace Dataset with a "text" column
        self.corpus = corpus
        self.q = BertQueryEncoder(device=self.device)

    # ────────────────────────────────────────────────────────────────
    #  forward: queries → (docs, sims)
    # ────────────────────────────────────────────────────────────────
    def forward(self, queries, k: int = 1):
        """
        queries : list[str]            batch of natural-language queries
        k       : int                  top-k passages to return
        Returns:
            docs : list[list[dict]]    retrieved corpus records
            sims : torch.Tensor        similarity scores (batch, k)
        """
        # 1. encode queries (batch, dim)  on self.device
        q_emb = self.q(queries)                       # torch tensor
        q_emb = F.normalize(q_emb, p=2, dim=-1)       # cosine simil.

        # 2. FAISS needs NumPy float32 on CPU
        q_np = q_emb.detach().cpu().numpy().astype("float32")
        D, I = self.index.search(q_np, k)             # (batch, k)

        # 3. similarities back to torch
        sims = torch.from_numpy(D).to(q_emb.device)

        # 4. map indices → corpus docs
        docs = [[self.corpus[int(idx)] for idx in row] for row in I]

        # optional: if we built the index on-the-fly from .pt embeddings,
        # recompute D via cosine sim for consistency
        if hasattr(self, "embeddings"):
            I_pt = torch.from_numpy(I).to(torch.long)
            emb_vectors = self.embeddings[I_pt]       # (batch,k,dim)
            q_expanded  = q_emb.unsqueeze(1)
            sims = F.cosine_similarity(q_expanded, emb_vectors, dim=-1)

        # convert to probabilities
        probs = F.softmax(sims, dim=-1)
        return docs, probs
