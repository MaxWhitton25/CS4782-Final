from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from model import EndtoEndRAG

EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-5

print("Loading dataset...")
qa_pairs = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")
corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")["passages"]
print(qa_pairs)
full_dataset = qa_pairs["test"]  # use entire dataset
vd_path = "Embeddings/bioasq_passage_embeddings.pt"

# Split into 80% train / 20% test
split = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
test_dataset = split["test"]


def my_collate_fn(examples):
    default_collator = DefaultDataCollator(return_tensors="pt")
    questions = [ex["question"] for ex in examples]
    answers = [ex["answer"] for ex in examples]
    batch = default_collator(examples)
    batch["question"] = questions
    batch["answer"] = answers
    return batch


train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate_fn
)

test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate_fn
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EndtoEndRAG(vd_path=vd_path, corpus=corpus, device=device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
model.to(device)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs, scores = model(batch["question"], batch["answer"])

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs, scores = model(batch["question"], batch["answer"])
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)


# Training loop
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # Train
    train_loss = train_epoch(model, train_dataloader, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")

    # Evaluate
    val_loss = evaluate(model, test_dataloader, device)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")

print("Training complete!")
