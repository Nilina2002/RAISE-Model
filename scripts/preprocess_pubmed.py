import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Folders
RAW_FOLDER = "data/raw/pubmed_xml/"
OUTPUT_FOLDER = "data/processed_chunks/pubmed_chunks/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to parse a single XML file
def parse_pubmed_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    chunks = []
    
    for article in root.findall('.//PubmedArticle'):
        title_elem = article.find('.//ArticleTitle')
        abstract_elem = article.find('.//Abstract/AbstractText')
        
        if title_elem is not None and abstract_elem is not None:
            title = title_elem.text
            abstract = abstract_elem.text
            if abstract:
                # Combine title + abstract
                text = f"{title}. {abstract}"
                chunks.append(text)
    return chunks

# Process all XML files
all_chunks = []
xml_files = [f for f in os.listdir(RAW_FOLDER) if f.endswith(".xml")]

for xml_file in tqdm(xml_files):
    file_path = os.path.join(RAW_FOLDER, xml_file)
    chunks = parse_pubmed_xml(file_path)
    all_chunks.extend(chunks)

# Save each chunk as a text file
for i, chunk in enumerate(all_chunks, 1):
    out_file = os.path.join(OUTPUT_FOLDER, f"chunk_{i:05d}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(chunk)

print(f"Processed {len(all_chunks)} PubMed chunks.")
