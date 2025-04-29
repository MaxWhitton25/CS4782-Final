import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from datasets import Dataset
from generator.generator import RAGGenerator
from retriever.retriever import Retriever

DOCUMENT_PATH = ""


class EndtoEndRAG(nn.Module):
    def __init__(self, vd_path: str, dataset: Dataset, device="cpu"):
        """
        Initializes the RAG model with a retriever and a generator.

        Args:
            vd_path (str): Path to the vector database (for retriever).
            document_path (str): Path to the document store.
            device (str): Device to run the models on ('cpu' or 'cuda').
        """
        super().__init__()
        self.retriever = Retriever(vd_path=vd_path, dataset=dataset, device=device)
        self.generator = RAGGenerator(device=device)

    def forward(self, query, k=5, device="cpu"):
        """
        Args:
            query (str): The input query.
            k (int): Number of documents to retrieve.
        Returns:
            List[str]: Generated answers for each retrieved document.
        """
        if not self.training:
            # Retrieve top-k documents for the query
            docs, _ = self.retriever(query, k)

            # Generate an answer for each (query, doc) pair
            answers = []
            for doc in docs:
                answer = self.generator.generate(question=query, context=doc)
                answers.append(answer)

            return answers
        else:
            overall_loss = 0

            DUMMY_QUESTION = " "
            DUMMY_TARGET = " "
            docs, doc_probs = self.retriever(query, k)
            losses = []
            for doc in docs:
                losses.append(
                    self.generator.train_run(
                        f"{DUMMY_QUESTION} {self.tokenizer.eos_token} {doc}",
                        DUMMY_TARGET,
                        device,
                    )
                )

            return overall_loss
