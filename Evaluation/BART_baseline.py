import sys
import os

# Add src directory to PYTHONPATH for generator import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from generator.generator import RAGGenerator
from Evaluate import evaluate_generation

##############RESULTS NO_CONTEXT#########################
#Average BLEU-1 : 0.1086
#Average ROUGE-L: 0.2225
#########################################################

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load full dataset (only 'test' split exists)
print("Loading dataset...")
ds = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")
full_dataset = ds["test"]  # use entire dataset

# Split into 80% train / 20% test
split = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
test_dataset  = split["test"]

# Collator for tensors
default_collator = DefaultDataCollator(return_tensors="pt")

# Custom collator to keep raw text
def my_collate_fn(examples):
    questions = [ex["question"] for ex in examples]
    answers   = [ex["answer"]   for ex in examples]
    batch = default_collator(examples)
    batch["question"] = questions
    batch["answer"]   = answers
    return batch

# DataLoaders
train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=my_collate_fn
)
test_dataloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=my_collate_fn
)

# Initialize model
print("Initializing the model...")
generator = RAGGenerator(model_name="facebook/bart-base", device=device)

# Evaluate baseline (no context)
all_results = []
all_scores = []

print("Evaluating on test split...")
for batch in tqdm(test_dataloader, total=len(test_dataloader)):
    for i, question in enumerate(batch["question"]):
        ref = batch["answer"][i]
        gen = generator.generate(question=question, context="")
        scores = evaluate_generation(ref, gen)
        all_results.append({
            "question": question,
            "reference_answer": ref,
            "generated_answer": gen,
            "bleu1": scores["BLEU-1"],
            "rougeL": scores["ROUGE-L"]
        })
        all_scores.append((scores["BLEU-1"], scores["ROUGE-L"]))

# Compute averages
avg_bleu  = sum(b for b, _ in all_scores) / len(all_scores)
avg_rouge = sum(r for _, r in all_scores) / len(all_scores)

print(f"\nAverage BLEU-1 : {avg_bleu:.4f}")
print(f"Average ROUGE-L: {avg_rouge:.4f}")

# Save results
pd.DataFrame(all_results).to_csv("bart_baseline_results.csv", index=False)

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist([b for b, _ in all_scores], bins=20, alpha=0.7)
plt.axvline(avg_bleu, color="r", linestyle="--"); plt.title("BLEU-1")
plt.subplot(1, 2, 2)
plt.hist([r for _, r in all_scores], bins=20, alpha=0.7)
plt.axvline(avg_rouge, color="r", linestyle="--"); plt.title("ROUGE-L")
plt.tight_layout()
plt.savefig("bart_baseline_score_distribution.png")
plt.close()

print("Results saved.")
