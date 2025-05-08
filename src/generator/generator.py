from transformers.models.bart import BartForConditionalGeneration, BartTokenizer
import torch
import torch.nn as nn
from typing import Optional, List
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput


class RAGGenerator(nn.Module):
    """
    RAG Generator module using BART BASE.
    """

    def __init__(
        self, model_name: str = "facebook/bart-base", device: Optional[str] = None
    ):
        """
        Initialize the BART model and tokenizer.
        Args:
            model_name (str): HuggingFace model name.
            device (str, optional): Device to use ('cuda' or 'cpu'). If None, auto-detect.
        """
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = self.model.to(self.device)

    def forward(
        self,
        query: List[str],
        doc_list: List[List[str]],
        labels: Optional[List[str]] = None,
        max_length: int = 512,
        num_beams: int = 4,
        **kwargs,
    ):
        """
        Forward pass through the generator.
        Args:
            query (List[str]): The input queries.
            doc_list (List[List[str]]): The retrieved context/documents for each query.
            labels (List[str], optional): The input labels.
            max_length (int): Max length of generated answer.
            num_beams (int): Beam search width.
            **kwargs: Additional arguments for BartForConditionalGeneration.
        Returns:
            ModelOutput: HuggingFace model output (includes loss if labels are provided).
        """
        input_texts = []
        for q, docs in zip(query, doc_list):
            for doc in docs:
                input_texts.append(f"Document: {doc} \n\nQuery: {q}")
        # Tokenize inputs
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        # Tokenize labels
        label_inputs = self.tokenizer(
            labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        repeats = torch.tensor([len(docs) for docs in doc_list], device=self.device)
        repeated_label_input_ids = torch.repeat_interleave(
            label_inputs["input_ids"], repeats, dim=0
        ).to(self.device)

        # Forward pass with labels for training
        outputs: Seq2SeqLMOutput = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=repeated_label_input_ids,
            return_dict=True,
        )

        return outputs

    def get_tokenizer(self):
        """
        Utility to get the matching tokenizer.
        """
        return self.tokenizer

    def generate(
        self,
        query: List[str],
        doc_list: List[List[str]],
        max_length: int = 512,
        num_beams: int = 4,
    ):
        """
        Generate an answer given a question and context.
        Args:
            query (List[str]): The input queries.
            doc_list (List[List[str]]): The retrieved context/documents for each query.
            max_length (int): Max length of generated answer.
            num_beams (int): Beam search width.
        """
        # Concatenate question and context as input
        input_texts = []
        for q, docs in zip(query, doc_list):
            for doc in docs:
                input_texts.append(f"{q} {self.tokenizer.eos_token} {doc}")
        # Tokenize inputs
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        # Forward pass without labels for inference
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            output_logits=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        return outputs
