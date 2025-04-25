import torch
import torch.nn.functional as F
import torch.nn as nn

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

    def forward(self, x):
        """
        
        """
        # TODO

class Generator(nn.Module):
    def __init__(self):
        """
        """
        # TODO

    def forward(self, x):
        """
        
        """
        # TODO