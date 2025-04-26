# Retrieval-Augmented Generation (in RAG-Sequence mode)

https://arxiv.org/pdf/2005.11401

### Dataset
https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq

### Plan
- set up dataset
  - encode document using pre-trained BERT and use FAISS for storage
- retriever - start with pre-trained BERT
- generator - start with pre-trained BART
- eval script
  - Retrieval Quality (internal analysis)	Top-K Recall
  - QA BLEU-1 and ROUGE-L
- training script
