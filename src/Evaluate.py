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
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import evaluate
from model import EndtoEndRAG



K_VALUE = 5
BATCH_SIZE = 4


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

vd_path = "Embeddings/bioasq_passage_embeddings.pt"
corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")["passages"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EndtoEndRAG(vd_path=vd_path, corpus=corpus, device=device)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))



tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")



# Internal variable to hold the ROUGE evaluator
_rouge = None

def _setup_metrics():
    """
    Internal setup function to initialize required resources.
    Called automatically on first use of evaluate_generation().
    """
    global _rouge
    if _rouge is None:
        nltk.download('punkt', quiet=True)
        _rouge = evaluate.load('rouge')

def evaluate_generation(reference, prediction):
    """
    Evaluates a predicted answer against the reference answer using BLEU-1 and ROUGE-L.
    Automatically initializes required resources on first use.

    Args:
        reference (str): Ground truth answer.
        prediction (str): Model-generated answer.

    Returns:
        dict: BLEU-1 score and ROUGE-L F1 score.
    """
    global _rouge
    if _rouge is None:
        _setup_metrics()

    # Tokenize for BLEU
    reference_tokens = nltk.word_tokenize(reference)
    prediction_tokens = nltk.word_tokenize(prediction)

    # BLEU-1: unigram weights
    smoothie = SmoothingFunction().method4
    bleu1_score = sentence_bleu(
        [reference_tokens],
        prediction_tokens,
        weights=(1, 0, 0, 0),  # BLEU-1
        smoothing_function=smoothie
    )

    # ROUGE-L
    rouge_result = _rouge.compute(predictions=[prediction], references=[reference])
    rouge_l_f1 = rouge_result['rougeL']

    return {
        'BLEU-1': bleu1_score,
        'ROUGE-L': rouge_l_f1
    }








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


test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate_fn
)





# Evaluate baseline (no context)
all_results = []
all_scores = []

print("Evaluating on test split...")
for batch in tqdm(test_dataloader, total=len(test_dataloader)):
    outputs, scores = model(batch["question"], batch["answer"], K_VALUE)
    predicted_ids = outputs.logits.argmax(dim=-1)
    decoded_output = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    
    for i, (question, decoded_output, ref) in enumerate(zip(batch["question"], decoded_output, batch["answer"])):
        scores = evaluate_generation(ref, decoded_output)

        # print(batch["question"])
        # print(batch["answer"])
        # Decode to string

        all_results.append({
            "question": question,
            "reference_answer": ref,
            "generated_answer": decoded_output,
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
