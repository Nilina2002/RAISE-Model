"""
RETRO Model with Multi-Hop Reasoning Integration
===============================================

This module extends the original RETRO model to integrate multi-hop reasoning
capabilities, enabling the model to perform iterative reasoning across multiple
knowledge chunks during generation.

Key Features:
- Multi-hop chunked cross-attention
- Iterative reasoning during generation
- Enhanced context integration
- Reasoning path tracking
- Backward compatibility with original RETRO

Usage:
    from scripts.retro_multihop_model import RETROMultiHopModel
    
    model = RETROMultiHopModel(vocab_size=30000)
    output = model.generate_with_reasoning(query, max_hops=3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import math
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import os

# Import our multi-hop reasoning components
from multihop_reasoning import MultiHopReasoningSystem, ReasoningPath, ReasoningStep
from enhanced_query_retro import EnhancedRetroQuery

# Import original RETRO components
from retro_model import (
    MultiHeadAttention, 
    ChunkedCrossAttention, 
    RestrictedAttention,
    RETROEncoderLayer, 
    RETRODecoderLayer,
    RETROModel
)


class MultiHopChunkedCrossAttention(nn.Module):
    """Enhanced chunked cross-attention with multi-hop reasoning capabilities."""
    
    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 64, 
                 max_hops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.max_hops = max_hops
        
        # Multi-hop attention layers
        self.hop_attentions = nn.ModuleList([
            MultiHeadAttention(d_model, n_heads, dropout)
            for _ in range(max_hops)
        ])
        
        # Hop combination layer
        self.hop_combiner = nn.Linear(d_model * max_hops, d_model)
        
        # Reasoning state tracking
        self.reasoning_weights = nn.Parameter(torch.ones(max_hops) / max_hops)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, retrieved_chunks: torch.Tensor,
                reasoning_path: Optional[ReasoningPath] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, d_model] - Input query sequence
            retrieved_chunks: [batch_size, num_chunks, chunk_size, d_model] - Retrieved chunks
            reasoning_path: Optional reasoning path for multi-hop attention
            mask: [batch_size, seq_len] - Optional mask for query
        """
        batch_size, seq_len, d_model = query.size()
        num_chunks = retrieved_chunks.size(1)
        
        # Reshape retrieved chunks for cross-attention
        retrieved_flat = retrieved_chunks.view(batch_size, -1, d_model)
        
        # Multi-hop attention processing
        hop_outputs = []
        
        for hop in range(self.max_hops):
            # Apply attention for this hop
            hop_output = self.hop_attentions[hop](
                query=query,
                key=retrieved_flat,
                value=retrieved_flat,
                mask=mask
            )
            hop_outputs.append(hop_output)
        
        # Combine hop outputs
        combined_hops = torch.cat(hop_outputs, dim=-1)
        attended_output = self.hop_combiner(combined_hops)
        
        # Apply reasoning weights if reasoning path is available
        if reasoning_path is not None:
            reasoning_weights = self._compute_reasoning_weights(reasoning_path)
            attended_output = attended_output * reasoning_weights.unsqueeze(0).unsqueeze(0)
        
        return self.dropout(attended_output)
    
    def _compute_reasoning_weights(self, reasoning_path: ReasoningPath) -> torch.Tensor:
        """Compute attention weights based on reasoning path confidence."""
        weights = torch.zeros(self.max_hops)
        
        for i, step in enumerate(reasoning_path.steps[:self.max_hops]):
            weights[i] = step.confidence
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(self.max_hops) / self.max_hops
        
        return weights


class MultiHopRETRODecoderLayer(nn.Module):
    """Enhanced decoder layer with multi-hop reasoning capabilities."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, chunk_size: int = 64,
                 window_size: int = 32, max_hops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.self_attention = RestrictedAttention(d_model, n_heads, window_size, dropout)
        self.multihop_cross_attention = MultiHopChunkedCrossAttention(
            d_model, n_heads, chunk_size, max_hops, dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, retrieved_chunks: torch.Tensor,
                reasoning_path: Optional[ReasoningPath] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with restricted attention
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Multi-hop chunked cross-attention
        cca_output = self.multihop_cross_attention(x, retrieved_chunks, reasoning_path, mask)
        x = self.norm2(x + self.dropout(cca_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class RETROMultiHopModel(nn.Module):
    """Enhanced RETRO model with multi-hop reasoning capabilities."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_encoder_layers: int = 6, n_decoder_layers: int = 6, d_ff: int = 2048,
                 chunk_size: int = 64, window_size: int = 32, max_seq_len: int = 512,
                 dropout: float = 0.1, num_retrieved_chunks: int = 5, max_hops: int = 3):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_retrieved_chunks = num_retrieved_chunks
        self.chunk_size = chunk_size
        self.max_hops = max_hops
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Encoder (same as original RETRO)
        self.encoder_layers = nn.ModuleList([
            RETROEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Enhanced decoder with multi-hop reasoning
        self.decoder_layers = nn.ModuleList([
            MultiHopRETRODecoderLayer(d_model, n_heads, d_ff, chunk_size, 
                                    window_size, max_hops, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Reasoning state tracking
        self.reasoning_state = nn.Parameter(torch.zeros(d_model))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def create_padding_mask(self, x: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """Create padding mask for input sequence."""
        return (x != pad_token_id).unsqueeze(1).unsqueeze(2).float()
    
    def forward(self, input_ids: torch.Tensor, retrieved_chunks: torch.Tensor,
                reasoning_path: Optional[ReasoningPath] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-hop RETRO model.
        
        Args:
            input_ids: [batch_size, seq_len] - Input token ids
            retrieved_chunks: [batch_size, num_chunks, chunk_size, d_model] - Retrieved chunk embeddings
            reasoning_path: Optional reasoning path for multi-hop attention
            attention_mask: [batch_size, seq_len] - Optional attention mask
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] - Output logits
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Add reasoning state if available
        if reasoning_path is not None:
            reasoning_emb = self.reasoning_state.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            x = x + reasoning_emb
        
        # Create padding mask
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        # Enhanced decoder with multi-hop reasoning
        for layer in self.decoder_layers:
            x = layer(x, retrieved_chunks, reasoning_path, attention_mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits


class RETROMultiHopSystem:
    """Complete system integrating multi-hop reasoning with RETRO model."""
    
    def __init__(self, model_path: str = None, corpus_path: str = None,
                 index_path: str = None, embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_hops: int = 3):
        # Get project paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        self.corpus_path = corpus_path or os.path.join(project_root, "data", "processed_chunks", "full_corpus.pkl")
        self.index_path = index_path or os.path.join(project_root, "faiss_index", "medical_faiss.index")
        self.max_hops = max_hops
        
        # Load retrieval system
        self.load_retrieval_system(embedder_model)
        
        # Initialize enhanced query system
        self.query_system = EnhancedRetroQuery(self.corpus_path, self.index_path, embedder_model)
        
        # Initialize multi-hop RETRO model
        self.retro_model = None
        if model_path and os.path.exists(model_path):
            self.load_retro_model(model_path)
        else:
            # Create new model
            self.retro_model = RETROMultiHopModel(
                vocab_size=30000,
                max_hops=max_hops
            )
    
    def load_retrieval_system(self, embedder_model: str):
        """Load the existing retrieval system components."""
        print(f"Loading multi-hop RETRO system with model: {embedder_model}")
        
        # Load corpus
        with open(self.corpus_path, "rb") as f:
            self.full_corpus = pickle.load(f)
            
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load embedder
        self.embedder = SentenceTransformer(embedder_model)
        
        print(f"Multi-hop RETRO system loaded. Corpus size: {len(self.full_corpus)}")
    
    def load_retro_model(self, model_path: str):
        """Load a pre-trained multi-hop RETRO model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.retro_model = RETROMultiHopModel(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            n_heads=checkpoint['n_heads'],
            n_encoder_layers=checkpoint['n_encoder_layers'],
            n_decoder_layers=checkpoint['n_decoder_layers'],
            d_ff=checkpoint['d_ff'],
            chunk_size=checkpoint['chunk_size'],
            window_size=checkpoint['window_size'],
            max_hops=checkpoint.get('max_hops', self.max_hops)
        )
        self.retro_model.load_state_dict(checkpoint['model_state_dict'])
        self.retro_model.eval()
        print(f"Multi-hop RETRO model loaded from {model_path}")
    
    def generate_with_reasoning(self, query: str, max_length: int = 100, 
                              max_hops: int = None) -> Dict:
        """Generate response using multi-hop reasoning."""
        if self.retro_model is None:
            raise ValueError("RETRO model not loaded. Please load a model first.")
        
        if max_hops is None:
            max_hops = self.max_hops
        
        print(f"Generating response with multi-hop reasoning (max_hops={max_hops})")
        
        # Perform multi-hop reasoning
        reasoning_result = self.query_system.reason(query, max_hops)
        reasoning_path = reasoning_result.reasoning_path
        
        if reasoning_path is None:
            raise ValueError("Failed to generate reasoning path")
        
        # Encode chunks from reasoning path
        all_chunks = []
        for step in reasoning_path.steps:
            all_chunks.extend(step.retrieved_chunks)
        
        chunk_embeddings = self._encode_chunks(all_chunks)
        
        # Simple tokenization (in practice, use proper tokenizer)
        input_tokens = query.split()[:50]  # Limit input length
        input_ids = torch.tensor([[hash(token) % 30000 for token in input_tokens]], dtype=torch.long)
        
        # Generate response with reasoning path
        with torch.no_grad():
            logits = self.retro_model(input_ids, chunk_embeddings, reasoning_path)
            # Simple generation - in practice, use proper generation methods
            output_tokens = torch.argmax(logits, dim=-1)
        
        # Convert back to text (simplified)
        response = " ".join([f"token_{token.item()}" for token in output_tokens[0]])
        
        return {
            'query': query,
            'response': response,
            'reasoning_path': reasoning_path,
            'reasoning_steps': len(reasoning_path.steps),
            'confidence': reasoning_path.confidence,
            'chunks_used': len(all_chunks),
            'processing_time': reasoning_result.processing_time
        }
    
    def _encode_chunks(self, chunks: List[str]) -> torch.Tensor:
        """Encode retrieved chunks into embeddings."""
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        
        # Convert to tensor and reshape for RETRO model
        batch_size = 1
        num_chunks = len(chunks)
        embedding_dim = embeddings.shape[1]
        
        # Pad or truncate to chunk_size
        chunk_size = min(len(max(chunks, key=len)), 64)  # Use actual max chunk length or 64
        
        # Create tensor with proper shape
        chunk_embeddings = torch.zeros(batch_size, num_chunks, chunk_size, embedding_dim)
        
        for i, chunk in enumerate(chunks):
            # Tokenize and embed chunk (simplified - in practice you'd use proper tokenization)
            chunk_tokens = chunk.split()[:chunk_size]  # Simple word splitting
            if len(chunk_tokens) < chunk_size:
                # Pad with zeros
                chunk_tokens.extend([''] * (chunk_size - len(chunk_tokens)))
            
            # Encode each token position
            for j, token in enumerate(chunk_tokens[:chunk_size]):
                if token:
                    token_emb = self.embedder.encode([token], convert_to_numpy=True)
                    chunk_embeddings[0, i, j, :] = torch.from_numpy(token_emb[0])
        
        return chunk_embeddings
    
    def compare_generation_approaches(self, query: str) -> Dict:
        """Compare single-hop vs multi-hop generation approaches."""
        print(f"Comparing generation approaches for: '{query}'")
        
        # Single-hop generation (original RETRO behavior)
        single_hop_result = self.query_system.retrieve(query, 5)
        single_hop_chunks = single_hop_result.chunks
        single_hop_embeddings = self._encode_chunks(single_hop_chunks)
        
        # Simple tokenization
        input_tokens = query.split()[:50]
        input_ids = torch.tensor([[hash(token) % 30000 for token in input_tokens]], dtype=torch.long)
        
        # Single-hop generation
        with torch.no_grad():
            single_hop_logits = self.retro_model(input_ids, single_hop_embeddings)
            single_hop_tokens = torch.argmax(single_hop_logits, dim=-1)
        
        single_hop_response = " ".join([f"token_{token.item()}" for token in single_hop_tokens[0]])
        
        # Multi-hop generation
        multi_hop_result = self.generate_with_reasoning(query)
        
        return {
            'query': query,
            'single_hop': {
                'response': single_hop_response,
                'chunks_used': len(single_hop_chunks),
                'confidence': single_hop_result.confidence
            },
            'multi_hop': {
                'response': multi_hop_result['response'],
                'chunks_used': multi_hop_result['chunks_used'],
                'confidence': multi_hop_result['confidence'],
                'reasoning_steps': multi_hop_result['reasoning_steps']
            }
        }


def create_multihop_retro_model(vocab_size: int = 30000, d_model: int = 512, 
                               n_heads: int = 8, n_encoder_layers: int = 6,
                               n_decoder_layers: int = 6, d_ff: int = 2048,
                               chunk_size: int = 64, window_size: int = 32,
                               max_hops: int = 3) -> RETROMultiHopModel:
    """Factory function to create a multi-hop RETRO model."""
    return RETROMultiHopModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        chunk_size=chunk_size,
        window_size=window_size,
        max_hops=max_hops
    )


if __name__ == "__main__":
    # Example usage
    print("Creating Multi-Hop RETRO model...")
    model = create_multihop_retro_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example with multi-hop system
    print("\nInitializing Multi-Hop RETRO system...")
    multihop_system = RETROMultiHopSystem()
    
    # Test multi-hop generation
    query = "What are the treatment options for diabetes and how do they relate to blood pressure management?"
    result = multihop_system.generate_with_reasoning(query)
    
    print(f"\nMulti-hop generation result:")
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Reasoning steps: {result['reasoning_steps']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Chunks used: {result['chunks_used']}")
    
    # Compare approaches
    comparison = multihop_system.compare_generation_approaches(query)
    print(f"\nComparison:")
    print(f"Single-hop chunks: {comparison['single_hop']['chunks_used']}")
    print(f"Multi-hop chunks: {comparison['multi_hop']['chunks_used']}")
    print(f"Multi-hop reasoning steps: {comparison['multi_hop']['reasoning_steps']}")
