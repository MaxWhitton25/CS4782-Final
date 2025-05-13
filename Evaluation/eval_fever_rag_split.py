#!/usr/bin/env python3
"""
Evaluation/eval_fever_rag_split.py
──────────────────────────────────────
Enhanced evaluation script for FEVER fact verification.
By default, evaluates the fine-tuned RAG model on different dataset splits.

Key options
-----------
--eval_split
    Which split to evaluate on: 'custom_test' (default), 'labelled_dev', 
    'paper_dev', or other available splits
--ckpt       Path to checkpoint  (default: src/Training/batched_checkpoints/fever_rag_split.pt)
--indices    Path to JSON with test-row indices (for custom_test mode)
--batch_size Mini-batch size     (default: 32)
--device     cuda / cpu          (auto-detect default)
--top_k      Retrieval depth     (default: 1)
--csv_out    Output CSV filename (default based on eval_split)
--exact_n    Keep only rows whose id appears *exactly* n times in test set
--exclude_seen_claims
    If passed, drop every test row whose claim *id* is present in the
    training portion. This guarantees no train→test leakage.
--compare_with_base_bart evaluate against BART model as baseline
"""

import argparse, json, sys, csv, os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import Counter

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]   # repo root
SRC  = REPO / "src"
sys.path.insert(0, str(SRC))

from verifier.verifier import RAGVerifier
from retriever.retriever import Retriever

FAISS_PATH = REPO / "Embeddings/fever_pages.faiss"
IDS_JSON   = REPO / "Embeddings/fever_pages_ids.json"
DPR_CONFIG = "psgs_w100.nq.exact"

LABEL_TOK  = {
    "SUPPORTS":         "<supports>",
    "REFUTES":          "<refutes>",
    "NOT ENOUGH INFO":  "<unknown>",
}
INV_LABEL  = {v: k for k, v in LABEL_TOK.items()}
ALLOWED    = set(LABEL_TOK.values())

# Available splits from error message
AVAILABLE_SPLITS = ['train', 'labelled_dev', 'unlabelled_dev', 'unlabelled_test', 'paper_dev', 'paper_test']

def build_corpus() -> Dataset:
    with open(IDS_JSON) as f:
        keep = set(json.load(f))
    stream = load_dataset("wiki_dpr", DPR_CONFIG, split="train", streaming=True)

    texts = []
    for ex in tqdm(stream, total=len(keep), desc="Collect DPR passages"):
        if ex["id"] in keep:
            texts.append(ex["text"])
            if len(texts) == len(keep):
                break
    return Dataset.from_dict({"text": texts})

def init_bart_baseline(device: str):
    print("Loading base BART model for comparison...")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model.to(device)
    model.eval()
    return model, tokenizer

def classify_batch_with_bart(
    model: BartForConditionalGeneration, 
    tokenizer: BartTokenizer,
    claims: List[str], 
    ctxs: List[str], 
    device: str
) -> List[str]:
    """Generate classification labels from the vanilla BART model."""
    # Format inputs for the vanilla BART model (prompt engineering)
    formatted_inputs = []
    for claim, ctx in zip(claims, ctxs):
        # Create a prompt that frames the task as claim verification
        if ctx:
            prompt = f"Verify the claim: '{claim}' based on this evidence: '{ctx}'. The claim is"
        else:
            prompt = f"Verify the claim: '{claim}' without any evidence. The claim is"
        formatted_inputs.append(prompt)
    
    # Tokenize inputs
    inputs = tokenizer(
        formatted_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=10,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode generated text
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Map generated text to FEVER labels
    result = []
    for text in decoded:
        text = text.strip().lower()
        if "support" in text or "true" in text or "correct" in text:
            result.append("<supports>")
        elif "refute" in text or "false" in text or "wrong" in text or "incorrect" in text:
            result.append("<refutes>")
        else:
            result.append("<unknown>")
            
    return result

def classify_batch(verifier, claims: List[str], ctxs: List[str], device: str) -> List[str]:
    tok = verifier.tokenizer
    eos = tok.eos_token
    enc = tok(
        [f"{c} {eos} {ctx}" for c, ctx in zip(claims, ctxs)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outs = verifier.model.generate(
            **enc, max_length=4, num_beams=1, early_stopping=True
        )

    decoded = tok.batch_decode(outs, skip_special_tokens=False)

    def to_label(text: str) -> str:
        text = text.strip()
        for cand in ALLOWED:
            if cand in text:
                return cand
        return "<unknown>"

    return [to_label(t) for t in decoded]

def get_available_splits():
    """Get a list of available splits from the FEVER dataset"""
    try:
        dataset = load_dataset("fever", "v1.0")
        return list(dataset.keys())
    except Exception:
        # If we can't load the dataset, return the list from the error message
        return AVAILABLE_SPLITS

def load_dataset_split(args) -> Tuple[Dataset, Optional[Set[int]]]:
    """
    Loads the specified dataset split for evaluation.
    Returns the dataset and optionally a set of training IDs (if needed for filtering).
    """
    train_ids = None
    
    if args.eval_split == "custom_test":
        # 1. Load the custom test set using indices
        print("🗂  Loading FEVER train split with custom test indices...")
        fever_train = load_dataset("fever", "v1.0", split="train")

        if not Path(args.indices).exists():
            raise FileNotFoundError(f"Test indices file not found: {args.indices}")
            
        with open(args.indices) as f:
            test_idx = json.load(f)
        test_ds = fever_train.select(test_idx)
        
        # Get training IDs if needed for filtering
        if args.exclude_seen_claims:
            train_idx = [i for i in range(len(fever_train)) if i not in test_idx]
            train_ids = {fever_train[i]["id"] for i in train_idx}
    else:
        # 2. Load any official split directly
        print(f"Loading FEVER '{args.eval_split}' split...")
        try:
            test_ds = load_dataset("fever", "v1.0", split=args.eval_split)
        except ValueError as e:
            available = get_available_splits()
            print(f"Error: Split '{args.eval_split}' not found.")
            print(f"Available splits: {available}")
            print(f"Original error: {e}")
            sys.exit(1)
            
    print(f"loaded {len(test_ds):,} examples for evaluation")
    return test_ds, train_ids

def get_labels_from_batch(batch: Dict) -> List[str]:
    """
    Extract labels from a batch, handling different field names and formats.
    """
    # Check different possible label field names
    if "label" in batch:
        # String labels
        if isinstance(batch["label"][0], str):
            return [LABEL_TOK[g] if g in LABEL_TOK else "<unknown>" for g in batch["label"]]
        # Integer labels (mapping needed)
        elif isinstance(batch["label"][0], (int, np.integer)):
            # FEVER label mapping is usually: 0=SUPPORTS, 1=REFUTES, 2=NOT ENOUGH INFO
            mapping = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
            return [LABEL_TOK[mapping[g]] for g in batch["label"]]
    
    # Try alternative field names
    elif "labels" in batch:
        if isinstance(batch["labels"][0], str):
            return [LABEL_TOK[g] if g in LABEL_TOK else "<unknown>" for g in batch["labels"]]
        elif isinstance(batch["labels"][0], (int, np.integer)):
            mapping = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
            return [LABEL_TOK[mapping[g]] for g in batch["labels"]]
    
    # In case of unlabelled data or missing labels
    print("No label field found in batch. Using placeholder labels.")
    return ["<unknown>"] * len(batch["claim"])

def main() -> None:
    # Get available splits
    available_splits = get_available_splits() + ["custom_test"]
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      default="src/Training/batched_checkpoints/fever_rag_split.pt")
    ap.add_argument("--indices",   default="src/Training/batched_checkpoints/fever_rag_split_test_indices.json")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--top_k",      type=int, default=1)
    ap.add_argument("--csv_out",    default=None,
                    help="Output CSV filename (default based on eval_split)")
    ap.add_argument("--exact_n",    type=int, default=None,
                    help="If set, keep only test rows whose claim id occurs exactly n times inside the test subset")
    ap.add_argument("--exclude_seen_claims", action="store_true",
                    help="Remove test rows whose claim id appears in the training portion")
    ap.add_argument("--eval_split", default="custom_test", choices=available_splits,
                    help=f"Which dataset split to evaluate on. Available: {', '.join(available_splits)}")
    ap.add_argument("--compare_with_base_bart", action="store_true",
                    help="Compare with a non-finetuned BART model as baseline")
    args = ap.parse_args()

    ckpt_path = (REPO / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    
    # Default CSV name based on eval split if not provided
    if args.csv_out is None:
        args.csv_out = f"preds_fever_{args.eval_split}.csv"
    
    csv_out_path = (REPO / "Evaluation" / args.csv_out) if not Path(args.csv_out).is_absolute() else Path(args.csv_out)
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)

    device  = args.device
    bs      = args.batch_size
    k       = args.top_k

    test_ds, train_ids = load_dataset_split(args)

    # optional: filter by exact_n occurrences inside *test* subset
    if args.exact_n is not None and "id" in test_ds.features:
        n = args.exact_n
        id_counts = Counter(test_ds["id"])
        keep_ids = {i for i, c in id_counts.items() if c == n}
        test_ds  = test_ds.filter(lambda ex: ex["id"] in keep_ids)
        print(f"kept rows with id occurring exactly {n}× → {len(test_ds):,}")

    # optional: exclude ids seen in training
    if args.exclude_seen_claims and train_ids is not None and "id" in test_ds.features:
        before = len(test_ds)
        test_ds = test_ds.filter(lambda ex: ex["id"] not in train_ids)
        after  = len(test_ds)
        print(f"filtered unseen-claim test rows: {after:,} (dropped {before-after:,})")

    print("Building corpus and initializing retriever...")
    corpus_ds = build_corpus()
    retriever = Retriever(str(FAISS_PATH), corpus=corpus_ds, device=device)

    print(f"Loading fine-tuned model from checkpoint: {ckpt_path}")
    verifier = RAGVerifier(device=device)
    ckpt_state = torch.load(ckpt_path, map_location=device)
    verifier.load_state_dict(ckpt_state["verifier_state"])
    retriever.q.load_state_dict(ckpt_state["query_encoder_state"])
    verifier.eval(); retriever.q.eval()
    
    bart_model = None
    bart_tokenizer = None
    if args.compare_with_base_bart:
        bart_model, bart_tokenizer = init_bart_baseline(device)

    print(f"Starting evaluation on {len(test_ds):,} examples...")
    y_true = [] 
    y_rag = []  
    y_base = []  
    
    # For BART baseline (if requested)
    y_bart = [] if args.compare_with_base_bart else None
    y_bart_no_ctx = [] if args.compare_with_base_bart else None
    
    for i in tqdm(range(0, len(test_ds), bs), desc="Evaluating"):
        batch = test_ds[i : i + bs]
        claims = batch["claim"]
        
        # Handle different label formats across splits
        gold = get_labels_from_batch(batch)

        # Retrieve contexts
        docs, _ = retriever(claims, k=k)
        ctxs = [" ".join(d["text"] for d in doc_list) for doc_list in docs]

        # Fine-tuned model predictions
        y_rag.extend(classify_batch(verifier, claims, ctxs, device))
        y_base.extend(classify_batch(verifier, claims, [""] * len(claims), device))
        
        # BART baseline predictions (if requested)
        if args.compare_with_base_bart:
            y_bart.extend(classify_batch_with_bart(bart_model, bart_tokenizer, claims, ctxs, device))
            y_bart_no_ctx.extend(classify_batch_with_bart(bart_model, bart_tokenizer, claims, [""] * len(claims), device))
        
        # Ground truth
        y_true.extend(gold)

    def report(name, preds):
        acc = accuracy_score(y_true, preds)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, preds, labels=list(ALLOWED), average="macro", zero_division=0
        )
        print(f"\n── {name} ")
        print(f" accuracy        : {acc:.4f}")
        print(f" macro-precision : {p:.4f}")
        print(f" macro-recall    : {r:.4f}")
        print(f" macro-F1        : {f:.4f}")
        print(
            classification_report(
                y_true,
                preds,
                labels=list(ALLOWED),
                target_names=[INV_LABEL[t] for t in ALLOWED],
                digits=3,
                zero_division=0,
            )
        )
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f}

    print(f"\nEvaluation Results ({args.eval_split} split):")
    metrics = {}
    metrics["rag"] = report(f"RAG (k={k})", y_rag)
    metrics["base"] = report("Baseline (no context)", y_base)
    
    if args.compare_with_base_bart:
        metrics["bart"] = report("BART Baseline (with context)", y_bart)
        metrics["bart_no_ctx"] = report("BART Baseline (no context)", y_bart_no_ctx)

    with csv_out_path.open("w", newline="", encoding="utf-8") as f:
        if args.compare_with_base_bart:
            w = csv.writer(f)
            w.writerow([
                "row_id", "claim", "gold", 
                "finetuned_rag", "finetuned_no_ctx",
                "bart_with_ctx", "bart_no_ctx"
            ])
            for rid, (ex, g, rp, bp, bc, bnc) in enumerate(zip(
                test_ds, y_true, y_rag, y_base, y_bart, y_bart_no_ctx
            )):
                w.writerow([
                    rid, ex["claim"], INV_LABEL.get(g, "UNKNOWN"), 
                    INV_LABEL[rp], INV_LABEL[bp],
                    INV_LABEL[bc], INV_LABEL[bnc]
                ])
        else:
            w = csv.writer(f)
            w.writerow(["row_id", "claim", "gold", "rag_pred", "base_pred"])
            for rid, (ex, g, rp, bp) in enumerate(zip(test_ds, y_true, y_rag, y_base)):
                w.writerow([rid, ex["claim"], INV_LABEL.get(g, "UNKNOWN"), INV_LABEL[rp], INV_LABEL[bp]])
    
    print(f"\nPredictions written to {csv_out_path}")
    
    summary_path = csv_out_path.with_suffix(".summary.json")
    summary = {
        "args": vars(args),
        "metrics": metrics,
        "dataset_info": {
            "split": args.eval_split,
            "size": len(test_ds),
        }
    }
    
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary metrics written to {summary_path}")

if __name__ == "__main__":
    main()
