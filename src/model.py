import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
import numpy as np
from datasets import load_dataset
from generator.generator import RAGGenerator
from retriever.retriever import Retriever

VD_PATH = ""
DOCUMENT_PATH = ""


class EndtoEndRAG(nn.Module):
    def __init__(self, vd_path=VD_PATH, document_path=DOCUMENT_PATH, device="cpu"):
        """
        Initializes the RAG model with a retriever and a generator.

        Args:
            vd_path (str): Path to the vector database (for retriever).
            document_path (str): Path to the document store.
            device (str): Device to run the models on ('cpu' or 'cuda').
        """
        super().__init__()
        self.retriever = Retriever(
            vd_path=vd_path, document_path=document_path, device=device
        )
        self.generator = RAGGenerator(device=device)

    def forward(self, query, k=5):
        """
        Args:
            query (str): The input query.
            k (int): Number of documents to retrieve.
        Returns:
            List[str]: Generated answers for each retrieved document.
        """
        # Retrieve top-k documents for the query
        docs = self.retriever(query, k)

        # Generate an answer for each (query, doc) pair
        answers = []
        for doc in docs:
            answer = self.generator.generate(question=query, context=doc)
            answers.append(answer)

        return answers
