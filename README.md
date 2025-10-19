# RETRO Model - Medical Knowledge Retrieval System

A retrieval-augmented generation (RETRO) model implementation for medical knowledge retrieval using FAISS indexing and sentence transformers. This project processes medical literature from PubMed and medical dictionaries to create a searchable knowledge base for medical queries.

## 🏥 Project Overview

This project implements a RETRO (Retrieval-Enhanced Transformer) model specifically designed for medical knowledge retrieval. It combines:

- **PubMed XML data**: Medical research abstracts and titles
- **Medical Dictionary**: Comprehensive medical terminology definitions
- **FAISS Indexing**: Efficient similarity search using Facebook's FAISS library
- **Sentence Transformers**: State-of-the-art text embeddings for semantic search

## 📁 Project Structure

```
raise-model/
├── data/
│   ├── raw/
│   │   ├── medical_dictionary/
│   │   │   └── DictionaryofMedicalTerms.pdf
│   │   └── pubmed_xml/
│   │       └── pubmed25n0001.xml
│   └── processed_chunks/
│       ├── dictionary_chunks/     # 477 processed dictionary entries
│       ├── pubmed_chunks/         # 15,377 processed PubMed abstracts
│       └── full_corpus.pkl        # Combined corpus for indexing
├── faiss_index/
│   └── medical_faiss.index        # FAISS index for fast retrieval
├── scripts/
│   ├── preprocess_dictionary.py   # Extract text from medical dictionary PDF
│   ├── preprocess_pubmed.py       # Parse PubMed XML files
│   ├── combine_chunks.py          # Combine all text chunks
│   ├── build_faiss_index.py       # Create FAISS index from corpus
│   └── query_retro.py             # Query the retrieval system
├── notebooks/                     # Jupyter notebooks for analysis
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- 8GB+ RAM (for processing large datasets)
- 2GB+ disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd raise-model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place medical dictionary PDF files in `data/raw/medical_dictionary/`
   - Place PubMed XML files in `data/raw/pubmed_xml/`

### Usage

#### 1. Preprocess Data

**Extract medical dictionary entries:**
```bash
python scripts/preprocess_dictionary.py
```

**Parse PubMed XML files:**
```bash
python scripts/preprocess_pubmed.py
```

**Combine all text chunks:**
```bash
python scripts/combine_chunks.py
```

#### 2. Build FAISS Index

```bash
python scripts/build_faiss_index.py
```

This will:
- Load the combined corpus (15,854+ text chunks)
- Generate embeddings using sentence transformers
- Build a FAISS HNSW index for fast similarity search
- Save the index to `faiss_index/medical_faiss.index`

#### 3. Query the System

```bash
python scripts/query_retro.py
```

Or use the retrieval function in your own code:

```python
from scripts.query_retro import retrieve

# Query the medical knowledge base
results = retrieve("What are the symptoms of diabetes?", top_k=5)
for i, chunk in enumerate(results, 1):
    print(f"Result {i}: {chunk[:200]}...")
```

## 🔧 Configuration

### Embedding Models

The system supports multiple sentence transformer models. Edit `scripts/build_faiss_index.py` to change the model:

```python
# Available models:
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, general purpose
# model_name = "allenai/scibert_scivocab_uncased"      # Scientific text
# model_name = "dmis-lab/biobert-base-cased-v1.1"      # Medical text (if available)
# model_name = "sentence-transformers/all-mpnet-base-v2" # Higher quality
```

### FAISS Index Configuration

The system uses HNSW (Hierarchical Navigable Small World) indexing for fast approximate nearest neighbor search:

```python
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node
```

## 📊 Dataset Statistics

- **Dictionary Chunks**: 477 medical term definitions
- **PubMed Chunks**: 15,377 research abstracts
- **Total Corpus**: 15,854+ text chunks
- **Index Size**: ~2GB (depending on embedding model)

## 🎯 Use Cases

This RETRO model is designed for:

- **Medical Research**: Find relevant research papers and definitions
- **Clinical Decision Support**: Retrieve medical knowledge for patient care
- **Medical Education**: Access comprehensive medical terminology
- **Literature Review**: Efficiently search through medical literature
- **Question Answering**: Build medical Q&A systems

## 🔍 Example Queries

```python
# Medical conditions
retrieve("What are the symptoms of hypertension?")

# Treatment options
retrieve("How is diabetes treated?")

# Medical procedures
retrieve("What is a coronary angioplasty?")

# Drug information
retrieve("Side effects of metformin")

# Research topics
retrieve("Recent advances in cancer immunotherapy")
```

## 🛠️ Technical Details

### Architecture

1. **Data Processing Pipeline**:
   - PDF text extraction (medical dictionary)
   - XML parsing (PubMed abstracts)
   - Text chunking and cleaning
   - Corpus combination

2. **Embedding Generation**:
   - Sentence transformer models for semantic embeddings
   - Batch processing for efficiency
   - Float32 precision for FAISS compatibility

3. **Indexing**:
   - FAISS HNSW index for sub-linear search complexity
   - Persistent storage for reuse
   - Memory-efficient vector storage

4. **Retrieval**:
   - Query embedding generation
   - Top-k similarity search
   - Ranked result return

### Performance

- **Index Build Time**: ~10-30 minutes (depending on model)
- **Query Time**: <100ms for top-5 results
- **Memory Usage**: ~4-8GB during indexing
- **Storage**: ~2GB for index + embeddings

## 📈 Future Enhancements

- [ ] Support for more embedding models (BioBERT, ClinicalBERT)
- [ ] Query expansion and refinement
- [ ] Multi-modal retrieval (text + images)
- [ ] Real-time index updates
- [ ] Web interface for queries
- [ ] Evaluation metrics and benchmarks
- [ ] Integration with medical ontologies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FAISS**: Facebook AI Similarity Search for efficient vector search
- **Sentence Transformers**: Hugging Face for pre-trained embedding models
- **PubMed**: National Library of Medicine for medical literature data
- **RETRO Model**: DeepMind's retrieval-augmented generation approach

## 📞 Support

For questions, issues, or contributions, please:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Note**: This project is for educational and research purposes. Always consult qualified medical professionals for medical advice and decisions.
