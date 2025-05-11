import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertQueryEncoder(nn.Module):
    def __init__(self, pretrained_bert_model_name="bert-base-uncased", device = None):
        super(BertQueryEncoder, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained(pretrained_bert_model_name).to(self.device)

    def forward(self, text):
        tokenized_inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
        bert_output = self.bert_model(**tokenized_inputs)

        cls_embeddings = bert_output.last_hidden_state[:, 0, :].squeeze().cpu()
        return cls_embeddings

# torch.save(model.state_dict(), "bert_query_encoder.pth")
# model.load_state_dict(torch.load("bert_query_encoder.pth"))

        
        
