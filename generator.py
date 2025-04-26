from transformers.models.bart import BartForConditionalGeneration, BartTokenizer
import torch
import torch.nn as nn
from typing import Optional


class RAGGenerator(nn.Module):
    """
    RAG Generator module using BART BASE, suitable for integration in larger models.
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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass through the generator.
        Args:
            input_ids (torch.Tensor): Tokenized input ids.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Target token ids for teacher forcing.
            **kwargs: Additional arguments for BartForConditionalGeneration.
        Returns:
            ModelOutput: HuggingFace model output (includes loss if labels are provided).
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    @staticmethod
    def get_tokenizer(model_name: str = "facebook/bart-base"):
        """
        Utility to get the matching tokenizer.
        """
        return BartTokenizer.from_pretrained(model_name)

    def generate(
        self, question: str, context: str, max_length: int = 64, num_beams: int = 4
    ) -> str:
        """
        Generate an answer given a question and context.
        Args:
            question (str): The input question.
            context (str): The retrieved context/document.
            max_length (int): Max length of generated answer.
            num_beams (int): Beam search width.
        Returns:
            str: Generated answer.
        """
        # Concatenate question and context as input
        input_text = f"{question} {self.tokenizer.eos_token} {context}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
