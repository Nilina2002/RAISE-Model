import os
import pickle

# Get the project root directory (parent of scripts folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

processed_folders = [
    os.path.join(project_root, "data", "processed_chunks", "pubmed_chunks"),
    os.path.join(project_root, "data", "processed_chunks", "dictionary_chunks"),
]

full_corpus = []

for folder in processed_folders:
    if os.path.exists(folder):
        files = sorted(os.listdir(folder))
        for file_name in files:
            file_path = os.path.join(folder, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    full_corpus.append(text)
    else:
        print(f"Warning: Folder not found: {folder}")

print(f"Total chunks combined: {len(full_corpus)}")

# Save the full corpus for later
output_dir = os.path.join(project_root, "data", "processed_chunks")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "full_corpus.pkl")
with open(output_file, "wb") as f:
    pickle.dump(full_corpus, f)

print(f"Full corpus saved to: {output_file}")
