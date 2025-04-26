import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
import numpy as np
from datasets import load_dataset
from generator import Generator
from retriever import Retriever

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

