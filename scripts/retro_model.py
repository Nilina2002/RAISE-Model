import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import os


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with optional masking."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(context)


class ChunkedCrossAttention(nn.Module):
    """Chunked Cross-Attention (CCA) mechanism for RETRO model."""
    
    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
    def forward(self, query: torch.Tensor, retrieved_chunks: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, d_model] - Input query sequence
            retrieved_chunks: [batch_size, num_chunks, chunk_size, d_model] - Retrieved chunks
            mask: [batch_size, seq_len] - Optional mask for query
        """
        batch_size, seq_len, d_model = query.size()
        num_chunks = retrieved_chunks.size(1)
        
        # Reshape retrieved chunks for cross-attention
        # [batch_size, num_chunks, chunk_size, d_model] -> [batch_size, num_chunks * chunk_size, d_model]
        retrieved_flat = retrieved_chunks.view(batch_size, -1, d_model)
        
        # Apply cross-attention
        attended_output = self.attention(
            query=query,
            key=retrieved_flat,
            value=retrieved_flat,
            mask=mask
        )
        
        return attended_output


class RestrictedAttention(nn.Module):
    """Restricted encoder-decoder attention with locality bias."""
    
    def __init__(self, d_model: int, n_heads: int, window_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
    def create_restricted_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a mask that restricts attention to nearby positions."""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
            
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = query.size()
        
        # Create restricted attention mask
        restricted_mask = self.create_restricted_mask(seq_len, query.device)
        
        # Combine with provided mask if any
        if mask is not None:
            # Ensure mask has correct dimensions and convert to boolean
            if mask.dim() == 4:  # Already has batch and head dimensions
                mask = mask.squeeze(1)  # Remove head dimension temporarily
            if mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            elif mask.dim() == 2:  # [batch, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, -1, -1)
            # Convert to boolean for bitwise operations
            mask = mask.bool()
            restricted_mask = restricted_mask.bool() & mask
            
        return self.attention(query, key, value, restricted_mask)


class RETROEncoderLayer(nn.Module):
    """Single encoder layer for RETRO model."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class RETRODecoderLayer(nn.Module):
    """Single decoder layer for RETRO model with chunked cross-attention."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, chunk_size: int = 64, 
                 window_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.self_attention = RestrictedAttention(d_model, n_heads, window_size, dropout)
        self.chunked_cross_attention = ChunkedCrossAttention(d_model, n_heads, chunk_size, dropout)
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
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with restricted attention
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Chunked cross-attention with retrieved chunks
        cca_output = self.chunked_cross_attention(x, retrieved_chunks, mask)
        x = self.norm2(x + self.dropout(cca_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class RETROModel(nn.Module):
    """Main RETRO model implementation."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_encoder_layers: int = 6, n_decoder_layers: int = 6, d_ff: int = 2048,
                 chunk_size: int = 64, window_size: int = 32, max_seq_len: int = 512,
                 dropout: float = 0.1, num_retrieved_chunks: int = 5):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_retrieved_chunks = num_retrieved_chunks
        self.chunk_size = chunk_size
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            RETROEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            RETRODecoderLayer(d_model, n_heads, d_ff, chunk_size, window_size, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
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
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of RETRO model.
        
        Args:
            input_ids: [batch_size, seq_len] - Input token ids
            retrieved_chunks: [batch_size, num_chunks, chunk_size, d_model] - Retrieved chunk embeddings
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
        
        # Create padding mask
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
            
        # Decoder with chunked cross-attention
        for layer in self.decoder_layers:
            x = layer(x, retrieved_chunks, attention_mask)
            
        # Output projection
        logits = self.output_projection(x)
        
        return logits


class RETRORetrievalSystem:
    """Integration class for RETRO model with existing retrieval system."""
    
    def __init__(self, model_path: str = None, corpus_path: str = None, 
                 index_path: str = None, embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Get project paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        self.corpus_path = corpus_path or os.path.join(project_root, "data", "processed_chunks", "full_corpus.pkl")
        self.index_path = index_path or os.path.join(project_root, "faiss_index", "medical_faiss.index")
        
        # Load retrieval system
        self.load_retrieval_system(embedder_model)
        
        # Initialize RETRO model
        self.retro_model = None
        if model_path and os.path.exists(model_path):
            self.load_retro_model(model_path)
            
    def load_retrieval_system(self, embedder_model: str):
        """Load the existing retrieval system components."""
        print(f"Loading retrieval system with model: {embedder_model}")
        
        # Load corpus
        with open(self.corpus_path, "rb") as f:
            self.full_corpus = pickle.load(f)
            
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load embedder
        self.embedder = SentenceTransformer(embedder_model)
        
        print(f"Retrieval system loaded. Corpus size: {len(self.full_corpus)}")
        
    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant chunks for a query."""
        query_emb = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_emb, top_k)
        return [self.full_corpus[i] for i in indices[0]]
        
    def encode_chunks(self, chunks: List[str]) -> torch.Tensor:
        """Encode retrieved chunks into embeddings."""
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        # Convert to tensor and reshape for RETRO model
        # [num_chunks, embedding_dim] -> [1, num_chunks, chunk_size, d_model]
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
        
    def load_retro_model(self, model_path: str):
        """Load a pre-trained RETRO model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.retro_model = RETROModel(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            n_heads=checkpoint['n_heads'],
            n_encoder_layers=checkpoint['n_encoder_layers'],
            n_decoder_layers=checkpoint['n_decoder_layers'],
            d_ff=checkpoint['d_ff'],
            chunk_size=checkpoint['chunk_size'],
            window_size=checkpoint['window_size']
        )
        self.retro_model.load_state_dict(checkpoint['model_state_dict'])
        self.retro_model.eval()
        print(f"RETRO model loaded from {model_path}")
        
    def generate_response(self, query: str, max_length: int = 100) -> str:
        """Generate a response using the RETRO model with retrieved chunks."""
        if self.retro_model is None:
            raise ValueError("RETRO model not loaded. Please load a model first.")
            
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_chunks(query, self.retro_model.num_retrieved_chunks)
        
        # Encode chunks
        chunk_embeddings = self.encode_chunks(retrieved_chunks)
        
        # Simple tokenization (in practice, use proper tokenizer)
        input_tokens = query.split()[:50]  # Limit input length
        
        # Convert to tensor (simplified - use proper tokenizer)
        input_ids = torch.tensor([[hash(token) % 30000 for token in input_tokens]], dtype=torch.long)
        
        # Generate response
        with torch.no_grad():
            logits = self.retro_model(input_ids, chunk_embeddings)
            # Simple generation - in practice, use proper generation methods
            output_tokens = torch.argmax(logits, dim=-1)
            
        # Convert back to text (simplified)
        response = " ".join([f"token_{token.item()}" for token in output_tokens[0]])
        
        return response


def create_retro_model(vocab_size: int = 30000, d_model: int = 512, n_heads: int = 8,
                      n_encoder_layers: int = 6, n_decoder_layers: int = 6, d_ff: int = 2048,
                      chunk_size: int = 64, window_size: int = 32) -> RETROModel:
    """Factory function to create a RETRO model with specified parameters."""
    return RETROModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        chunk_size=chunk_size,
        window_size=window_size
    )


if __name__ == "__main__":
    # Example usage
    print("Creating RETRO model...")
    model = create_retro_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example with retrieval system
    print("\nInitializing RETRO retrieval system...")
    retro_system = RETRORetrievalSystem()
    
    # Test retrieval
    query = "What are the symptoms of diabetes?"
    chunks = retro_system.retrieve_chunks(query, top_k=3)
    print(f"\nRetrieved {len(chunks)} chunks for query: '{query}'")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:100]}...")
