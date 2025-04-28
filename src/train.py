from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from model import EndtoEndRAG


EPOCHS = 5

print("Loading dataset...")
ds = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")
full_dataset = ds["test"]  # use entire dataset

# Split into 80% train / 20% test
split = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
test_dataset  = split["test"]

def my_collate_fn(examples):
    default_collator = DefaultDataCollator(return_tensors="pt")

    questions = [ex["question"] for ex in examples]
    answers   = [ex["answer"]   for ex in examples]
    batch = default_collator(examples)
    batch["question"] = questions
    batch["answer"]   = answers
    return batch

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    collate_fn=my_collate_fn
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=True, 
    collate_fn=my_collate_fn
)


model = EndtoEndRAG()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        
        # Forward pass
        outputs, scores = model(batch["question"])  
        logits = outputs.logits  
        
        loss = loss_fn(logits, batch["answer"])  
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
torch.save(model.state_dict(), "model.pth")