from transformers.models.bart import BartForConditionalGeneration, BartTokenizer
import torch
import torch.nn as nn
from typing import Optional


class RAGVerifier(nn.Module):
    """
    RAG Verifier module using BART BASE for fact verification (classification) tasks.
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

        # Add label tokens for classification
        special_tokens = {"additional_special_tokens": ["<supports>", "<refutes>", "<unknown>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

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
        Forward pass through the verifier.
        Args:
            input_ids (torch.Tensor): Tokenized input ids.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Target token ids (single label token).
            **kwargs: Additional arguments for BartForConditionalGeneration.
        Returns:
            ModelOutput: HuggingFace model output (includes loss if labels are provided).
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def get_tokenizer(self):
        """
        Utility to get the matching tokenizer.
        """
        return self.tokenizer

    def train_run(self, claim_text, context_text, target_label, device):
        """
        One training step: given a claim and context, predict the correct label.
        Args:
            claim_text (str): The claim to verify.
            context_text (str): The retrieved evidence/context.
            target_label (str): Target label ("<supports>", "<refutes>", or "<unknown>").
            device (str): Device to use.
        Returns:
            torch.Tensor: Loss value for this training step.
        """
        input_text = f"{claim_text} {self.tokenizer.eos_token} {context_text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        labels = self.tokenizer(
            target_label,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        label_ids = labels["input_ids"].to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_ids,
        )
        loss = outputs.loss
        return loss

    def classify(
        self, claim: str, context: str, num_beams: int = 1
    ) -> str:
        """
        Classify a claim given context into one of the label tokens.
        Args:
            claim (str): The claim to verify.
            context (str): The evidence/context.
            num_beams (int): Beam search width (default 1 for classification).
        Returns:
            str: Predicted label ("<supports>", "<refutes>", or "<unknown>").
        """
        input_text = f"{claim} {self.tokenizer.eos_token} {context}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_length=2,  # Predict a single token + EOS
            num_beams=num_beams,
            early_stopping=True,
        )
        prediction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return prediction
