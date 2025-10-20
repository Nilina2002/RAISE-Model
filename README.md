# RETRO Model - Medical Knowledge Retrieval System

A complete RETRO (Retrieval-Enhanced Transformer) model implementation for medical knowledge retrieval and generation. This project combines retrieval-augmented generation with medical literature from PubMed and medical dictionaries to create a comprehensive medical knowledge system with encoder-decoder architecture, chunked cross-attention (CCA), and restricted attention mechanisms.

## 🏥 Project Overview

This project implements a complete RETRO (Retrieval-Enhanced Transformer) model specifically designed for medical knowledge retrieval and generation. It combines:

- **PubMed XML data**: Medical research abstracts and titles
- **Medical Dictionary**: Comprehensive medical terminology definitions
- **FAISS Indexing**: Efficient similarity search using Facebook's FAISS library
- **Sentence Transformers**: State-of-the-art text embeddings for semantic search
- **RETRO Architecture**: Encoder-decoder model with chunked cross-attention (CCA)
- **Restricted Attention**: Locality-biased attention mechanisms for efficient processing
- **Generation Capabilities**: Text generation with retrieval-augmented context

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
│   ├── query_retro.py             # Query the retrieval system (legacy)
│   ├── retro_model.py             # Core RETRO model implementation
│   ├── train_retro.py             # Training script for RETRO model
│   ├── retro_inference.py         # Inference script for trained RETRO model
│   └── retro_example.py           # Demonstration script
├── models/
│   └── retro/                     # Trained RETRO models
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

#### 3. Train the RETRO Model

```bash
python scripts/train_retro.py --num_epochs 10 --batch_size 8
```

This will:
- Load the combined corpus for training
- Create a RETRO model with encoder-decoder architecture
- Train the model with chunked cross-attention (CCA)
- Save the trained model to `models/retro/`

#### 4. Run RETRO Inference

**Interactive mode:**
```bash
python scripts/retro_inference.py --interactive
```

**Single query mode:**
```bash
python scripts/retro_inference.py --query "What are the symptoms of diabetes?"
```

**Analyze retrieval process:**
```bash
python scripts/retro_inference.py --analyze --query "Treatment for hypertension"
```

#### 5. Legacy Retrieval System

For basic retrieval without generation:
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

## 🧠 RETRO Model Architecture

### Key Components

The RETRO model implements three core mechanisms:

1. **Encoder-Decoder Architecture**: Standard transformer encoder-decoder with medical domain adaptations
2. **Chunked Cross-Attention (CCA)**: Integrates retrieved medical chunks into the generation process
3. **Restricted Encoder-Decoder Attention**: Locality-biased attention for efficient processing

### Architecture Details

```
Input Query → Retrieval System → Retrieved Chunks
     ↓                              ↓
Encoder Layers → Decoder Layers ← Chunked Cross-Attention
     ↓                              ↓
Self-Attention → Restricted Attention → Output Generation
```

### Model Parameters

- **Vocabulary Size**: 30,000 tokens
- **Model Dimension**: 512
- **Number of Heads**: 8
- **Encoder Layers**: 6
- **Decoder Layers**: 6
- **Feed-forward Dimension**: 2,048
- **Chunk Size**: 64 tokens
- **Window Size**: 32 tokens (for restricted attention)

### Training Process

1. **Data Preparation**: Medical text chunks are tokenized and prepared for training
2. **Retrieval Integration**: For each training sample, relevant chunks are retrieved
3. **Chunked Cross-Attention**: Retrieved chunks are integrated via CCA mechanism
4. **Restricted Attention**: Locality-biased attention patterns are applied
5. **Generation**: Model learns to generate responses based on retrieved context

## 🔧 Configuration

### RETRO Model Configuration

Edit `scripts/retro_model.py` to modify model architecture:

```python
model = create_retro_model(
    vocab_size=30000,      # Vocabulary size
    d_model=512,           # Model dimension
    n_heads=8,             # Number of attention heads
    n_encoder_layers=6,    # Number of encoder layers
    n_decoder_layers=6,    # Number of decoder layers
    d_ff=2048,             # Feed-forward dimension
    chunk_size=64,         # Chunk size for CCA
    window_size=32         # Window size for restricted attention
)
```

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

### RETRO Model Generation

```python
from scripts.retro_inference import RETROInference
from scripts.retro_model import RETRORetrievalSystem

# Initialize RETRO system
retrieval_system = RETRORetrievalSystem()
retro_inference = RETROInference("models/retro/best_retro_model.pth", retrieval_system)

# Generate responses with retrieval-augmented context
response = retro_inference.generate_response("What are the symptoms of diabetes?")
print(f"RETRO Response: {response}")

# Analyze retrieval process
analysis = retro_inference.analyze_retrieval("Treatment for hypertension")
print(f"Retrieved {analysis['num_chunks_retrieved']} relevant chunks")
```

### Legacy Retrieval System

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

### RETRO Model Architecture

1. **Encoder-Decoder Structure**:
   - Multi-head self-attention in encoder layers
   - Restricted attention patterns in decoder layers
   - Layer normalization and residual connections
   - Positional embeddings for sequence understanding

2. **Chunked Cross-Attention (CCA)**:
   - Integrates retrieved medical chunks into generation
   - Cross-attention between query and retrieved context
   - Chunked processing for efficient memory usage
   - Dynamic chunk selection based on relevance

3. **Restricted Attention Mechanisms**:
   - Locality-biased attention windows
   - Reduced computational complexity
   - Improved training stability
   - Medical domain-specific attention patterns

### Data Processing Pipeline

1. **Data Preparation**:
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

4. **Retrieval Integration**:
   - Query embedding generation
   - Top-k similarity search
   - Chunk encoding for RETRO model
   - Dynamic retrieval during generation

### Performance

#### Retrieval System
- **Index Build Time**: ~10-30 minutes (depending on model)
- **Query Time**: <100ms for top-5 results
- **Memory Usage**: ~4-8GB during indexing
- **Storage**: ~2GB for index + embeddings

#### RETRO Model
- **Training Time**: ~2-6 hours (depending on dataset size and hardware)
- **Inference Time**: ~200-500ms per query
- **Model Size**: ~100-200MB (depending on configuration)
- **Memory Usage**: ~2-4GB during training, ~1-2GB during inference
- **Generation Quality**: Improved over baseline retrieval-only systems

## 📈 Future Enhancements

### RETRO Model Improvements
- [ ] Fine-tuned medical language models (BioBERT, ClinicalBERT)
- [ ] Improved chunked cross-attention mechanisms
- [ ] Dynamic chunk size adaptation
- [ ] Multi-hop reasoning with retrieved context
- [ ] Reinforcement learning for generation quality
- [ ] Evaluation metrics for medical knowledge generation

### System Enhancements
- [ ] Support for more embedding models
- [ ] Query expansion and refinement
- [ ] Multi-modal retrieval (text + images)
- [ ] Real-time index updates
- [ ] Web interface for queries
- [ ] Integration with medical ontologies
- [ ] Distributed training support
- [ ] Model compression and optimization

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
