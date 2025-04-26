from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import torch
from tqdm import tqdm

# Correct dataset loading with config name!
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

def embed_text(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)
    cls_embedding = output.last_hidden_state[:, 0, :].squeeze().cpu()
    return cls_embedding

# Prepare lists to store embeddings and corresponding IDs
embeddings = []
ids = []

for item in tqdm(train_data, desc="Generating embeddings"):
    text = item['passage']
    id_ = item['id']
    embedding = embed_text(text)
    embeddings.append(embedding)
    ids.append(id_)

# Stack embeddings into a single tensor
embedding_tensor = torch.stack(embeddings)

# Save embeddings and IDs
save_path = "bioasq_passage_embeddings.pt"
torch.save({
    "embeddings": embedding_tensor,
    "ids": ids
}, save_path)

print(f"Saved embeddings to {save_path}")
print(f"Embedding tensor shape: {embedding_tensor.shape}")
