from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from read_pdf import chunks  # importing chunks from your previous file

# -----------------------------
# Step 1: Load embedding model
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Step 2: Create embeddings
# -----------------------------
print("Creating embeddings for chunks...")
embeddings = model.encode(chunks, convert_to_numpy=True)

print(f"Total embeddings created: {len(embeddings)}")
print(f"Embedding vector size: {len(embeddings[0])}")

# -----------------------------
# Step 3: Create FAISS index
# -----------------------------
embedding_dim = embeddings.shape[1]  # should be 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(embedding_dim)  # simple L2 distance index
index.add(embeddings)

print(f"FAISS index contains {index.ntotal} vectors.")

# -----------------------------
# Step 4: Save the index
# -----------------------------
faiss.write_index(index, "faiss_index.bin")
# Optional: save chunks too (for retrieving text later)
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index and chunks saved successfully!")
