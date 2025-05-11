from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import torch
import numpy as np
from tqdm import tqdm

# Load dataset
dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")

# Check available splits
print(dataset)

train_data = dataset['passages']

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def embed_text(text: str, tokenizer: BertTokenizer, model: BertModel, device: torch.device) -> np.ndarray:
    """
    Compute CLS embedding for the given text using BERT.
    Returns a numpy array of shape (hidden_size,).
    """
    encoded_input = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=512,
    ).to(device)
    
    with torch.no_grad():
        output = model(**encoded_input)
    
    cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy().squeeze(0)
    return cls_embedding

# Prepare lists to store embeddings and corresponding IDs
embeddings = []
ids = []

for item in tqdm(train_data, desc="Generating embeddings"):
    text = item['passage']
    id_ = item['id']
    embedding = embed_text(text, tokenizer, model, device)  # Updated usage
    embeddings.append(embedding)
    ids.append(id_)

# Stack embeddings into a single NumPy array
embedding_array = np.stack(embeddings)

# Save embeddings and IDs
save_path = "bioasq_passage_embeddings.npz"
np.savez(save_path, embeddings=embedding_array, ids=np.array(ids))

print(f"Saved embeddings to {save_path}")
print(f"Embedding array shape: {embedding_array.shape}")
