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
        k: int = 2,
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
            outputs, generator_query_document_losses = self.generator.generate(query, docs, labels)
            generator_query_document_losses = generator_query_document_losses.view_as(doc_probs)

            #take logs and then add and then exponentiate to multiply without resulting in overflow
            log_doc_probs_minus_generator_query_document_losses = torch.exp(torch.log(doc_probs) - generator_query_document_losses)
            negative_log_likelihood_of_sum = -torch.log(torch.sum(log_doc_probs_minus_generator_query_document_losses, dim=-1))
            total_loss = torch.mean(negative_log_likelihood_of_sum)

            outputs.loss = total_loss

            return outputs, doc_probs
        else:
            # Generate outputs
            outputs = self.generator.generate(query, docs)

            return outputs, doc_probs
