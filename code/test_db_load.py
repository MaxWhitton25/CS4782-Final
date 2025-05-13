import torch
import numpy as np
import faiss

# Check if the file exists
index_path = "Embeddings/bioasq_passage_embeddings.pt"

# Load the embeddings
data = torch.load(index_path)
print("Successfully loaded embeddings")

embeddings = data["embeddings"]
ids = data["ids"]
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"First few IDs: {ids[:5]}")

# Convert to FAISS index
print("\nConverting to FAISS index...")
embeddings_np = embeddings.numpy().astype("float32")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# Save the FAISS index
faiss_index_path = "Embeddings/bioasq_passage_embeddings.faiss"
faiss.write_index(index, faiss_index_path)
print(f"Saved FAISS index to {faiss_index_path}")

# Test the index
test_vector = np.random.rand(1, dimension).astype("float32")
distances, indices = index.search(test_vector, 1)
print("\nFAISS test search successful")
print(f"Nearest neighbor distance: {distances[0][0]}")
print(f"Nearest neighbor index: {indices[0][0]}")

# Check if the saved index is loadable
try:
    loaded_index = faiss.read_index(faiss_index_path)
    print("\nSuccessfully loaded saved FAISS index")
except Exception as e:
    assert False, f"Failed to load saved FAISS index: {str(e)}"
