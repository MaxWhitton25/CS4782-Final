import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
import numpy as np

VD_PATH = ""
DOCUMENT_PATH = ""


class EndtoEndRAG(nn.Module):
    def __init__(self):
        """

        """

        ## TODO
        ################
        self.retriever = Retriever

        self.generator = Generator

       

    def forward(self, x):

        ## TODO
        #################

        z = self.retriever(x)

        out = self.generator(x, z)

        #################

        pass

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


    def forward(self, x, k):
        """
        
        """
        # TODO
        # This should be the embedding of the query:
        xq = np.random.random((1, self.d)).astype('float32')

        # I is the indices of the top k documents
        D, I = self.index.search(xq, k)  

        docs = get_docs_from_indices(I)

        return x + docs
        
def get_docs_from_indices(I, DOCUMENT_PATH):
    return ""


class Generator(nn.Module):
    def __init__(self):
        """
        """
        # TODO

    def forward(self, x):
        """
        
        """
        # TODO