from transformers.models.bart import BartForConditionalGeneration, BartTokenizer
import torch
import torch.nn as nn
from typing import Optional, List
from transformers.modeling_outputs import Seq2SeqLMOutput


class RAGGenerator(nn.Module):
    """
    RAG Generator module using BART.
    """

    def __init__(
        self, model_name: str = "facebook/bart-base", device: Optional[str] = None
    ):
        """
        Initialize the BART model and tokenizer.
        Args:
            model_name: HuggingFace model name. Should be a BART model.
            device: Device to use. If None, auto-detect.
        """
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = self.model.to(self.device)  # type: ignore

    def forward(
        self,
        query: List[str],
        doc_list: List[List[str]],
        labels: List[str],
        max_length: int = 512,
    ):
        """
        Forward pass through the generator. Note that forward is only used for training.
        Args:
            query: The input queries.
            doc_list: The retrieved context/documents for each query.
            labels: The input labels.
            max_length: Max length of generated answer.
        Returns:
            Seq2SeqLMOutput: HuggingFace model output (includes loss).
        """
        # Concatenate question and context as input
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

    def generate(
        self,
        query: List[str],
        doc_list: List[List[str]],
        max_length: int = 512,
        num_beams: int = 4,
    ):
        """
        Generate a response given a question and context. Note that generate is only used for inference.
        Args:
            query: The input queries.
            doc_list: The retrieved context/documents for each query.
            max_length: Max length of generated answer.
            num_beams: Beam search width.
        Returns:
            GenerateBeamEncoderDecoderOutput: The generated responses.
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
