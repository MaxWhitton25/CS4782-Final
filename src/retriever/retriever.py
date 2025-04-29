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
vd_path = "Embeddings\passage_embedding\bioasq_passage_embeddings.pt"

from QueryEncoder import BertQueryEncoder


class Retriever(nn.Module):
    def __init__(self, vd_path: str, dataset: Dataset, device: str = "cpu"):
        """
        Args:
            vd_path: path to the vector database - should be a faiss index
            dataset: dataset of documents
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
            embeddings = data["embeddings"]
            # Convert to FAISS index
            print("\nConverting to FAISS index...")
            embeddings_np = embeddings.numpy().astype("float32")
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_np)
            self.index = index
        else:
            raise ValueError(f"Unsupported file type: {vd_path.split('.')[-1]}")

        self.dataset = dataset

        self.q = BertQueryEncoder()

    def forward(self, x, k):
        """ """
        # TODO
        # This should be the embedding of the query:
        xq = self.q(x)
        print(xq.shape)

        # I is the indices of the top k documents
        D, I = self.index.search(xq.detach(), k)

        docs = [[self.dataset[int(i)] for i in row] for row in I]
        print(docs)
        return docs, probs
