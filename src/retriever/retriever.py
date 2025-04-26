import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
import numpy as np
from datasets import load_dataset

class Retriever(nn.Module):
    def __init__(self):
        """
        """
        # TODO
        # initializes database:
        xb = np.load(VD_PATH)
        self.d = xb.shape[1]
        self.index = faiss.IndexFlatL2(self.d)   
        print(self.index.is_trained)
        self.index.add(xb)                  
        print(self.index.ntotal)
        self.corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")['passages']


    def forward(self, x, k):
        """
        
        """
        # TODO
        # This should be the embedding of the query:
        xq = np.random.random((1, self.d)).astype('float32')

        # I is the indices of the top k documents
        D, I = self.index.search(xq, k)  

        docs = self.corpus[I]

        return x + docs



