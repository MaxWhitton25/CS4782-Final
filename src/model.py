import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from datasets import Dataset
from generator.generator import RAGGenerator
from retriever.retriever import Retriever
import nltk
from typing import List, Optional
import torch


class EndtoEndRAG(nn.Module):
    def __init__(self, vd_path: str, corpus: Dataset, device="cpu"):
        """
        Initializes the RAG model with a retriever and a generator.

        Args:
            vd_path (str): Path to the vector database (for retriever).
            corpus (Dataset): Dataset of passages.
            device (str): Device to run the models on ('cpu' or 'cuda').
        """
        super().__init__()
        self.retriever = Retriever(vd_path=vd_path, corpus=corpus, device=device)
        self.generator = RAGGenerator(device=device)
        self.tokenizer = nltk.word_tokenize

    def forward(
        self,
        query: List[str],
        labels: Optional[List[str]] = None,
        k: int = 1,
    ):
        """
        Args:
            query (List[str]): The input queries.
            labels (List[str], optional): The input labels.
            k (int): Number of documents to retrieve.
            device (str): Device to run the models on ('cpu' or 'cuda').
        """
        # Retrieve top-k documents for each query
        docs, doc_probs = self.retriever(query, k)

        if labels is not None:
            # Generate outputs
            outputs = self.generator.generate(query, docs, labels)

            # Get losses
            generator_loss = outputs.loss
            retriever_loss = -torch.log(doc_probs).mean()
            total_loss = generator_loss + retriever_loss

            outputs.loss = total_loss

            return outputs, doc_probs
        else:
            # Generate outputs
            outputs = self.generator.generate(query, docs)

            return outputs, doc_probs
