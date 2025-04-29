import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
import numpy as np
from datasets import load_dataset
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/retriever'))
vd_path = 'Embeddings\passage_embedding\bioasq_passage_embeddings.pt'

from QueryEncoder import BertQueryEncoder


class Retriever(nn.Module):
    def __init__(self, vd_path, document_path, device="cpu"):
        """ """
        # TODO: initialize the database and the query encoder
        super().__init__()

        data = torch.load('Embeddings\\passage_embedding\\bioasq_passage_embeddings.pt')
        tensor = data['embeddings'] 
        
        xb = tensor.cpu().numpy()
        self.d = xb.shape[1]
        self.index = faiss.IndexFlatL2(self.d)
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
        print(xq.shape)

        # I is the indices of the top k documents
        D, I = self.index.search(xq.detach(), k)
        D = torch.from_numpy(D)
        probs = F.softmax(D, dim=-1)
        docs = [[self.corpus[int(i)] for i in row] for row in I]
        print(docs)
        return docs, probs
    