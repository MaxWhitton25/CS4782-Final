# Retrieval-Augmented Generation (in RAG-Sequence mode)

[arXiv](https://arxiv.org/pdf/2005.11401)

### Dataset
[rag-mini-bioasq](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq)

### Poster
[poster.pdf](poster/poster.pdf)

# Introduction
This github repo is our reimplementation of a paper of our choice, the original RAG paper: Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks.

Large Language Models (LLMs) have proven to be effective tools across a wide range of natural language processing tasks. However, their ability to recall factual information is often limited by the fixed context window and static knowledge encoded in model parameters during pre-training. This can lead to gaps in model knowledge, especially on queries outside the scope of the training data. The paper reproduce via this repo introduced Retrieval-Augmented Generation (RAG), a technique that addresses this limitation by combining parametric knowledge from LLMs with non-parametric memory through document retrieval.

# Chosen Result

# Github Contents
* Code: includes our implementation, training loops, and evaluation pipelines for this project.
  * Code/generator: this folder includes our implementation of the generator module of RAG. It leverages BartForConditionalGeneration.
  * Code/retriever: this folder includes out implementation of the retriever module of RAG, including an encoder module (which uses BERT) and a retriever module that leverages the encoder and faiss to retrieve the correct documents.
* 
# Re-implementation Details

# Reproduction Steps

# Results / Insights

# Conclusion

# References

# Acknowledgements
