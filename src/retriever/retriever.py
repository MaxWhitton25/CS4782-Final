import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
import numpy as np
from datasets import load_dataset
from retriever.QueryEncoder import BertQueryEncoder


class Retriever(nn.Module):
    def __init__(self, vd_path, document_path, device="cpu"):
        """ """
        # TODO: initialize the database and the query encoder
        super().__init__()
        xb = np.load(vd_path)
        self.d = xb.shape[1]
        self.index = faiss.IndexFlatIP(self.d)
        print(self.index.is_trained)
        self.index.add(xb)
        print(self.index.ntotal)
        self.corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")[
            "passages"
        ]

        self.q = BertQueryEncoder()

    def forward(self, x, k):
        """ """
        # TODO
        # This should be the embedding of the query:
        xq = self.q(x)

        # I is the indices of the top k documents
        D, I = self.index.search(xq, k)
        probs = F.softmax(D, dim=-1)


        docs = self.corpus[I]

        return docs, probs
