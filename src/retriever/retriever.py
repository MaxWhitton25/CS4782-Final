import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
import numpy as np
from datasets import load_dataset
import sys
import os
from datasets import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/retriever"))

from QueryEncoder import BertQueryEncoder


class Retriever(nn.Module):
    def __init__(self, vd_path: str, corpus: Dataset, device: str = "cpu"):
        """
        Args:
            vd_path: path to the vector database - should be a faiss index
            corpus: dataset of passages
            device: device to run the model on
        """
        # TODO: initialize the database and the query encoder
        super().__init__()

        # Load the embeddings
        if vd_path.endswith(".faiss"):
            print("Loading FAISS index...")
            self.index = faiss.read_index(vd_path)
            print("Successfully loaded FAISS index")
            print(f"Index size: {self.index.ntotal}")
            print(f"Index dimension: {self.index.d}")
        elif vd_path.endswith(".pt"):
            data = torch.load(vd_path)
            self.embeddings = data["embeddings"]
            # Convert to FAISS index
            print("\nConverting to FAISS index...")
            embeddings_np = self.embeddings.numpy().astype("float32")
            dimension = self.embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_np)
            self.index = index
        else:
            raise ValueError(f"Unsupported file type: {vd_path.split('.')[-1]}")

        self.corpus = corpus

        self.q = BertQueryEncoder().to(device)

    def forward(self, x, k=1):
        """ """
        # TODO
        # This should be the embedding of the query:
        xq = self.q(x)

        # I is the indices of the top k documents
        D, I = self.index.search(xq.detach(), k)
        # #this should really always exist but dummy fix for now
        if hasattr(self, "embeddings"):
            I_pt = torch.from_numpy(I)
            embedding_vectors = self.embeddings[I_pt]
            xq_expanded = xq.unsqueeze(1)
            D = F.cosine_similarity(xq_expanded, embedding_vectors, dim=-1)

        probs = F.softmax(D, dim=-1)
        docs = [[self.corpus[int(i)] for i in row] for row in I]
        return docs, probs
