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
            outputs, generator_query_document_losses = self.generator.generate(
                query, docs, labels
            )
            generator_query_document_losses = generator_query_document_losses.view_as(
                doc_probs
            )

            # take logs and then add and then exponentiate to multiply without resulting in overflow
            log_doc_probs_minus_generator_query_document_losses = torch.exp(
                torch.log(doc_probs) - generator_query_document_losses
            )
            negative_log_likelihood_of_sum = -torch.log(
                torch.sum(log_doc_probs_minus_generator_query_document_losses, dim=-1)
            )
            total_loss = torch.mean(negative_log_likelihood_of_sum)

            outputs.loss = total_loss

            return outputs, doc_probs
        else:
            # Generate outputs
            outputs = self.generator.generate(query, docs)

            return outputs, doc_probs

    def generate(
        self,
        query: List[str],
        k: int = 2,
        max_length: int = 128,
        num_beams: int = 4,
    ) -> List[str]:
        """
        Implements RAG-Sequence with fast decoding.

        Args:
            query (List[str]): The input queries
            k (int): Number of documents to retrieve
            max_length (int): Maximum length of generated sequence
            num_beams (int): Number of beams for beam search

        Returns:
            List[str]: Generated sequences with highest marginalized probability
        """
        # Retrieve top-k documents
        docs, doc_probs = self.retriever(query, k)
        batch_size = len(query)

        final_outputs = []
        for batch_idx in range(batch_size):
            candidates = []
            candidate_scores = []

            # Generate independently for each document
            for doc_idx in range(k):
                # Get retrieval score
                retrieval_score = doc_probs[batch_idx][doc_idx]

                # Generate candidates using beam search for this document
                outputs = self.generator.generate(
                    query=[query[batch_idx]],
                    doc_list=[docs[batch_idx][doc_idx]],
                    max_length=max_length,
                    num_beams=num_beams,
                )

                # Get logits from generation output
                logits = outputs.logits

                # Calculate generation probability using the logits
                # TODO: Check this
                last_token_logits = logits[:, -1, :]
                generation_score = (
                    F.softmax(last_token_logits, dim=-1).max(dim=-1).values
                )

                total_score = retrieval_score * generation_score

                # Store candidate and its score
                candidates.append(outputs)
                candidate_scores.append(total_score)

            # Select candidate with highest marginalized probability
            best_idx = torch.argmax(torch.tensor(candidate_scores))
            best_candidate = candidates[best_idx]

            # Decode the best candidate using the generator's tokenizer
            # TODO: Check this
            final_output = self.generator.tokenizer.decode(
                best_candidate, skip_special_tokens=True
            )
            final_outputs.append(final_output)

        return final_outputs
