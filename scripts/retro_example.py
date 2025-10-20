"""
Example script demonstrating RETRO model usage.
This script shows how to use the RETRO model for medical knowledge retrieval and generation.
"""

import os
import torch
from retro_model import RETROModel, RETRORetrievalSystem, create_retro_model
from retro_inference import RETROInference


def demo_basic_retrieval():
    """Demonstrate basic retrieval functionality."""
    print("=" * 60)
    print("BASIC RETRIEVAL DEMONSTRATION")
    print("=" * 60)
    
    # Initialize retrieval system
    retrieval_system = RETRORetrievalSystem()
    
    # Example medical queries
    queries = [
        "What are the symptoms of Type 2 Diabetes?",
        "How is hypertension treated?",
        "What is coronary artery disease?",
        "Side effects of metformin",
        "Symptoms of pneumonia"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Retrieve relevant chunks
        chunks = retrieval_system.retrieve_chunks(query, top_k=3)
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}: {chunk[:150]}...")
            
    return retrieval_system


def demo_retro_model_creation():
    """Demonstrate RETRO model creation."""
    print("\n" + "=" * 60)
    print("RETRO MODEL CREATION DEMONSTRATION")
    print("=" * 60)
    
    # Create a RETRO model
    model = create_retro_model(
        vocab_size=30000,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        chunk_size=64,
        window_size=32
    )
    
    print(f"RETRO Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model architecture:")
    print(f"  - Vocabulary size: 30,000")
    print(f"  - Model dimension: 512")
    print(f"  - Number of heads: 8")
    print(f"  - Encoder layers: 6")
    print(f"  - Decoder layers: 6")
    print(f"  - Feed-forward dimension: 2,048")
    print(f"  - Chunk size: 64")
    print(f"  - Window size: 32")
    
    return model


def demo_chunked_cross_attention():
    """Demonstrate chunked cross-attention mechanism."""
    print("\n" + "=" * 60)
    print("CHUNKED CROSS-ATTENTION DEMONSTRATION")
    print("=" * 60)
    
    # Create a small model for demonstration
    model = create_retro_model(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=512,
        chunk_size=32,
        window_size=16
    )
    
    # Create sample data
    batch_size = 1
    seq_len = 20
    num_chunks = 3
    chunk_size = 32
    d_model = 128
    
    # Sample input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Sample retrieved chunks
    retrieved_chunks = torch.randn(batch_size, num_chunks, chunk_size, d_model)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Retrieved chunks shape: {retrieved_chunks.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, retrieved_chunks)
    
    print(f"Output logits shape: {logits.shape}")
    print("Chunked cross-attention mechanism working correctly!")
    
    return model


def demo_retrieval_integration():
    """Demonstrate integration with retrieval system."""
    print("\n" + "=" * 60)
    print("RETRIEVAL INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize retrieval system
    retrieval_system = RETRORetrievalSystem()
    
    # Create RETRO model
    model = create_retro_model(vocab_size=30000, d_model=512)
    
    # Example query
    query = "What are the treatment options for diabetes?"
    
    print(f"Query: {query}")
    
    # Retrieve chunks
    chunks = retrieval_system.retrieve_chunks(query, top_k=5)
    print(f"Retrieved {len(chunks)} chunks")
    
    # Encode chunks
    chunk_embeddings = retrieval_system.encode_chunks(chunks)
    print(f"Chunk embeddings shape: {chunk_embeddings.shape}")
    
    # Sample tokenization (in practice, use proper tokenizer)
    input_tokens = [1, 2, 3, 4, 5]  # Sample token IDs
    input_ids = torch.tensor([input_tokens], dtype=torch.long)
    
    print(f"Input token IDs: {input_tokens}")
    
    # Forward pass through RETRO model
    with torch.no_grad():
        logits = model(input_ids, chunk_embeddings)
    
    print(f"Output logits shape: {logits.shape}")
    print("RETRO model successfully integrated with retrieval system!")


def demo_attention_patterns():
    """Demonstrate different attention patterns in RETRO."""
    print("\n" + "=" * 60)
    print("ATTENTION PATTERNS DEMONSTRATION")
    print("=" * 60)
    
    from retro_model import MultiHeadAttention, RestrictedAttention, ChunkedCrossAttention
    
    d_model = 128
    n_heads = 4
    seq_len = 16
    chunk_size = 8
    window_size = 4
    
    # Sample data
    x = torch.randn(1, seq_len, d_model)
    
    print("1. Standard Multi-Head Attention:")
    standard_attn = MultiHeadAttention(d_model, n_heads)
    with torch.no_grad():
        output = standard_attn(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n2. Restricted Attention (with locality bias):")
    restricted_attn = RestrictedAttention(d_model, n_heads, window_size)
    with torch.no_grad():
        output = restricted_attn(x, x, x)
    print(f"   Window size: {window_size}")
    print(f"   Output shape: {output.shape}")
    
    print("\n3. Chunked Cross-Attention:")
    cca = ChunkedCrossAttention(d_model, n_heads, chunk_size)
    retrieved_chunks = torch.randn(1, 2, chunk_size, d_model)
    with torch.no_grad():
        output = cca(x, retrieved_chunks)
    print(f"   Query shape: {x.shape}")
    print(f"   Retrieved chunks shape: {retrieved_chunks.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\nAll attention mechanisms working correctly!")


def main():
    """Main demonstration function."""
    print("RETRO MODEL DEMONSTRATION")
    print("This script demonstrates the key components of the RETRO model implementation.")
    
    try:
        # 1. Basic retrieval demonstration
        retrieval_system = demo_basic_retrieval()
        
        # 2. RETRO model creation
        model = demo_retro_model_creation()
        
        # 3. Chunked cross-attention
        demo_chunked_cross_attention()
        
        # 4. Retrieval integration
        demo_retrieval_integration()
        
        # 5. Attention patterns
        demo_attention_patterns()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey features demonstrated:")
        print("✓ Basic retrieval functionality")
        print("✓ RETRO model architecture")
        print("✓ Chunked cross-attention (CCA)")
        print("✓ Restricted encoder-decoder attention")
        print("✓ Integration with existing retrieval system")
        print("\nNext steps:")
        print("1. Train the model using: python train_retro.py")
        print("2. Run inference using: python retro_inference.py --interactive")
        print("3. Analyze retrieval using: python retro_inference.py --analyze")
        
    except Exception as e:
        print(f"Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
