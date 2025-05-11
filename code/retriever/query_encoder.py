import torch
import torch.nn as nn
from transformers.models.bert import BertModel, BertTokenizer
from typing import Optional, List


class BertQueryEncoder(nn.Module):
    """
    BERT-based query encoder.
    """

    def __init__(
        self,
        pretrained_bert_model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
    ):
        """
        Initialize the BERT query encoder.
        Args:
            pretrained_bert_model_name: Name of the pretrained BERT model to use.
            device: Device to run the model on. If None, auto-detect.
        """
        super(BertQueryEncoder, self).__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained(pretrained_bert_model_name).to(
            self.device
        )

    def forward(self, text: List[str]):
        """
        Encode a list of text queries into a tensor of embeddings.
        Args:
            text: List of text queries to encode.
        Returns:
            torch.Tensor: Tensor of embeddings of shape (batch_size, embedding_dim).
        """
        tokenized_inputs = self.bert_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
        bert_output = self.bert_model(**tokenized_inputs)
        cls_embeddings = bert_output.last_hidden_state[:, 0, :].squeeze().cpu()
        return cls_embeddings
