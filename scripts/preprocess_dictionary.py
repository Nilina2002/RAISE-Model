import os
import pdfplumber

RAW_PDF_FOLDER = "data/raw/medical_dictionary/"
OUTPUT_FOLDER = "data/processed_chunks/dictionary_chunks/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pdf_files = [f for f in os.listdir(RAW_PDF_FOLDER) if f.endswith(".pdf")]

all_chunks = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(RAW_PDF_FOLDER, pdf_file)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Split paragraphs / definitions
                chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 20]
                all_chunks.extend(chunks)

# Save chunks
for i, chunk in enumerate(all_chunks, 1):
    out_file = os.path.join(OUTPUT_FOLDER, f"chunk_{i:05d}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(chunk)

print(f"Processed {len(all_chunks)} dictionary chunks.")
