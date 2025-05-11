import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
from typing import Optional, List
from datasets import Dataset
from .query_encoder import BertQueryEncoder


class Retriever(nn.Module):
    """
    Dense retriever using FAISS.
    """

    def __init__(self, vd_path: str, corpus: Dataset, device: Optional[str] = None):
        """
        Initializes the retriever with a vector database path and device.

        Args:
            vd_path: Path to the vector database.
            corpus: Corpus of passages.
            device: Device to use. If None, auto-detect.
        """
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if vd_path.endswith((".faiss", ".index")):
            # If FAISS index, load it
            print(f"Loading FAISS index from {vd_path}")
            self.index = faiss.read_index(vd_path)
            print(f"Loaded index: ntotal={self.index.ntotal}, dim={self.index.d}")
        elif vd_path.endswith(".pt"):
            # If embeddings, convert to FAISS index
            print(f"Loading embeddings from {vd_path}")
            data = torch.load(vd_path, map_location="cpu")
            emb = data["embeddings"]
            print(f"Converting embeddings to FAISS (dim={emb.shape[1]})")
            emb_np = emb.numpy().astype("float32")
            self.index = faiss.IndexFlatIP(emb_np.shape[1])
            self.index.add(emb_np)
        else:
            raise ValueError(f"Unsupported vector db type: {vd_path}")

        self.q_encoder = BertQueryEncoder(device=self.device)
        self.corpus = corpus

    def forward(self, queries: List[str], k: int = 1):
        """
        Retrieve top-k passages for a batch of input queries.

        Args:
            queries: batch of query strings
            k: number of top passages to return

        Returns:
            docs: List[List[record]] of shape (batch, k): top-k passages per query
            probs: torch.Tensor of shape (batch, k): retrieval probabilities
        """
        # Encode queries to embeddings on device
        q_emb = self.q_encoder(queries).to(self.device)
        q_emb = F.normalize(q_emb, p=2, dim=-1)

        # Convert to CPU numpy for FAISS and run Inner Product Search
        q_np = q_emb.detach().cpu().numpy().astype("float32")
        D, I = self.index.search(q_np, k)

        sims = F.normalize(torch.from_numpy(D).to(self.device), p=2, dim=-1)

        docs = [[self.corpus[int(idx)] for idx in row] for row in I]

        # Convert similarities to probabilities
        probs = F.softmax(sims, dim=-1)

        return docs, probs.to(self.device)
