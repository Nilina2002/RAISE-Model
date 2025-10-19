import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Get the directory of this script and construct the paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
corpus_path = os.path.join(project_root, "data", "processed_chunks", "full_corpus.pkl")
index_path = os.path.join(project_root, "faiss_index", "medical_faiss.index")

# Load corpus and index
with open(corpus_path, "rb") as f:
    full_corpus = pickle.load(f)

index = faiss.read_index(index_path)
# Using the same model as in build_faiss_index.py
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve(query, top_k=5):
    query_emb = embedder.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    return [full_corpus[i] for i in indices[0]]

# Example
query = "What are the common symptoms of Type 2 Diabetes?"
results = retrieve(query)
for i, chunk in enumerate(results, 1):
    print(f"Chunk {i}:", chunk[:300], "...\n")