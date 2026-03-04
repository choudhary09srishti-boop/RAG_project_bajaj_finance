import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# Step 1: Load FAISS index and chunks
# -----------------------------
index = faiss.read_index("faiss_index.bin")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# -----------------------------
# Step 2: Load embedding model
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Step 3: Search function
# -----------------------------
def search(query, top_k=3):
    query_vector = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, top_k)
    results = [chunks[i] for i in indices[0]]
    return results, distances[0]

# -----------------------------
# Step 4: Try a query
# -----------------------------
query = "loan interest rate"
results, distances = search(query)

for i, (chunk, dist) in enumerate(zip(results, distances)):
    print(f"\nResult {i+1} (Distance: {dist:.4f}):\n{chunk}")
