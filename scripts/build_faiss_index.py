import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# Get the directory of this script and construct the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
corpus_path = os.path.join(project_root, "data", "processed_chunks", "full_corpus.pkl")

# Load corpus
with open(corpus_path, "rb") as f:
    full_corpus = pickle.load(f)

print(f"Number of chunks: {len(full_corpus)}")

# Create embeddings
# Available models (uncomment the one you want to use):
# 1. General purpose (fast, good quality): "sentence-transformers/all-MiniLM-L6-v2"
# 2. Scientific text: "allenai/scibert_scivocab_uncased" 
# 3. Medical text: "dmis-lab/biobert-base-cased-v1.1" (if available)
# 4. Larger, higher quality: "sentence-transformers/all-mpnet-base-v2"

model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading model: {model_name}")
embedder = SentenceTransformer(model_name)
batch_size = 64
embeddings = []

for i in range(0, len(full_corpus), batch_size):
    batch = full_corpus[i:i+batch_size]
    batch_emb = embedder.encode(batch, convert_to_numpy=True)
    embeddings.append(batch_emb)

embeddings = np.vstack(embeddings).astype('float32')
print(f"Embeddings shape: {embeddings.shape}")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.add(embeddings)
print(f"FAISS index built. Total vectors: {index.ntotal}")

# Save FAISS index
faiss_index_dir = os.path.join(project_root, "faiss_index")
os.makedirs(faiss_index_dir, exist_ok=True)
faiss_index_path = os.path.join(faiss_index_dir, "medical_faiss.index")
faiss.write_index(index, faiss_index_path)
print(f"FAISS index saved to {faiss_index_path}")
