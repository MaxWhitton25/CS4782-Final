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
        labels: List[str],
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
        # Generate outputs
        outputs = self.generator(
            query=query,
            doc_list=docs,
            labels=labels,
        )
        generator_probs = F.softmax(
            outputs.logits, dim=-1
        )  # (batch_size*k, seq_len, vocab_size)

        # Reshape generator_probs to (batch_size, k, seq_len, vocab_size)
        batch_size = len(query)
        seq_len = generator_probs.size(1)
        vocab_size = generator_probs.size(2)
        generator_probs = generator_probs.view(batch_size, k, seq_len, vocab_size)

        # Convert to log space
        log_generator_probs = torch.log(generator_probs)

        # Calculate sequence log probabilities using log-sum-exp
        sequence_log_probs = torch.logsumexp(
            log_generator_probs, dim=-1
        )  # (batch_size, k, seq_len)

        # Calculate document-level log probabilities
        doc_level_log_probs = torch.sum(sequence_log_probs, dim=-1)  # (batch_size, k)

        # Convert doc_probs to log space
        log_doc_probs = torch.log(doc_probs)

        # Calculate final loss using log probabilities
        loss = -torch.logsumexp(log_doc_probs + doc_level_log_probs, dim=-1)

        return outputs, loss

    def generate(
        self,
        query: List[str],
        k: int = 2,
        max_length: int = 512,
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
                retrieval_score = torch.log(doc_probs[batch_idx][doc_idx])

                # Generate candidates using beam search for this document
                outputs = self.generator.generate(
                    query=[query[batch_idx]],
                    doc_list=[docs[batch_idx][doc_idx]],
                    max_length=max_length,
                    num_beams=num_beams,
                )

                # Get generation probabilities
                generated_ids = outputs.sequences[0]  # shape: (seq_len)
                # outputs.scores is a tuple of length sequence_length where each item is a tensor of size (batch_size * num_beams, vocab_size)
                generated_scores = outputs.scores

                generation_score = 0
                # Process each generation step
                for step, step_scores in enumerate(generated_scores):
                    if step + 1 >= len(generated_ids):
                        break

                    step_probs = F.softmax(step_scores, dim=-1)

                    token_id = generated_ids[step]
                    token_prob = step_probs[0, token_id]
                    # Compute total sequence probability in log space for numerical stability
                    generation_score += torch.log(token_prob)

                total_score = retrieval_score + generation_score

                candidates.append(outputs)
                candidate_scores.append(total_score.item())

            # Select candidate with highest marginalized probability
            best_idx = torch.argmax(torch.tensor(candidate_scores))
            best_candidate = candidates[best_idx]

            # Decode the best candidate
            final_output = self.generator.tokenizer.decode(
                best_candidate.sequences[0], skip_special_tokens=True
            )
            final_outputs.append(final_output)

        return final_outputs
