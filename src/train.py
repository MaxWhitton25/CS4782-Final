from model import EndtoEndRAG
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator


EPOCHS = 5

ds = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")

# Split the dataset into train and test sets
train_test_split = ds['test'].train_test_split(test_size=0.2)

# Access the train and test sets
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    collate_fn=DefaultDataCollator(return_tensors="pt")
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=True, 
    collate_fn=DefaultDataCollator(return_tensors="pt")
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
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch["question"])  
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