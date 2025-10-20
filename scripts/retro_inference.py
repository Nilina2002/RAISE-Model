import torch
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import argparse
from typing import List, Dict, Tuple
import json

from retro_model import RETROModel, RETRORetrievalSystem


class RETROInference:
    """Inference class for trained RETRO model."""
    
    def __init__(self, model_path: str, retrieval_system: RETRORetrievalSystem, device: str = 'cpu'):
        self.device = device
        self.retrieval_system = retrieval_system
        
        # Load trained model
        print(f"Loading RETRO model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        self.vocab = checkpoint['vocab']
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # Create model with loaded parameters
        self.model = RETROModel(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            n_heads=checkpoint['n_heads'],
            n_encoder_layers=checkpoint['n_encoder_layers'],
            n_decoder_layers=checkpoint['n_decoder_layers'],
            d_ff=checkpoint['d_ff'],
            chunk_size=checkpoint['chunk_size'],
            window_size=checkpoint['window_size']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print("RETRO model loaded successfully!")
        
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the model's vocabulary."""
        tokens = text.lower().split()
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return ' '.join([self.id_to_token.get(tid, '<unk>') for tid in token_ids])
        
    def generate_response(self, query: str, max_length: int = 100, 
                         temperature: float = 1.0, top_k: int = 50) -> str:
        """Generate a response using the RETRO model."""
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieval_system.retrieve_chunks(query, top_k=5)
        print(f"Retrieved {len(retrieved_chunks)} chunks for query: '{query}'")
        
        # Encode chunks
        chunk_embeddings = self.retrieval_system.encode_chunks(retrieved_chunks)
        chunk_embeddings = chunk_embeddings.to(self.device)
        
        # Tokenize query
        input_tokens = self.tokenize(query)
        input_tokens = [self.vocab['<sos>']] + input_tokens[:50]  # Limit input length
        
        # Convert to tensor
        input_ids = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
        
        # Generate response
        generated_tokens = input_tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get current sequence
                current_input = torch.tensor([generated_tokens], dtype=torch.long).to(self.device)
                
                # Create attention mask
                attention_mask = torch.ones_like(current_input)
                
                # Forward pass
                logits = self.model(current_input, chunk_embeddings, attention_mask)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop if EOS token
                if next_token == self.vocab['<eos>']:
                    break
                    
                generated_tokens.append(next_token)
                
        # Convert to text
        response = self.detokenize(generated_tokens[len(input_tokens):])
        return response
        
    def analyze_retrieval(self, query: str, top_k: int = 5) -> Dict:
        """Analyze the retrieval process for a query."""
        # Retrieve chunks
        retrieved_chunks = self.retrieval_system.retrieve_chunks(query, top_k)
        
        # Get retrieval scores
        query_emb = self.retrieval_system.embedder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.retrieval_system.index.search(query_emb, top_k)
        
        analysis = {
            'query': query,
            'num_chunks_retrieved': len(retrieved_chunks),
            'chunks': []
        }
        
        for i, (chunk, distance, idx) in enumerate(zip(retrieved_chunks, distances[0], indices[0])):
            analysis['chunks'].append({
                'rank': i + 1,
                'similarity_score': float(1.0 / (1.0 + distance)),  # Convert distance to similarity
                'corpus_index': int(idx),
                'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'length': len(chunk)
            })
            
        return analysis


def main():
    parser = argparse.ArgumentParser(description='RETRO Model Inference')
    parser.add_argument('--model_path', type=str, 
                       default='../models/retro/best_retro_model.pth',
                       help='Path to the trained RETRO model')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--query', type=str,
                       help='Single query to process')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum response length')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze retrieval process')
    
    args = parser.parse_args()
    
    # Get project paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_path = os.path.join(project_root, "models", "retro", "best_retro_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train_retro.py")
        return
        
    print("RETRO Model Inference")
    print("====================")
    print(f"Model path: {model_path}")
    print(f"Device: {args.device}")
    
    try:
        # Initialize retrieval system
        print("Initializing retrieval system...")
        retrieval_system = RETRORetrievalSystem()
        
        # Initialize inference
        print("Initializing RETRO inference...")
        retro_inference = RETROInference(model_path, retrieval_system, args.device)
        
        if args.interactive:
            # Interactive mode
            print("\nInteractive mode. Type 'quit' to exit.")
            print("=" * 50)
            
            while True:
                query = input("\nEnter your medical query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not query:
                    continue
                    
                print(f"\nProcessing query: '{query}'")
                print("-" * 30)
                
                if args.analyze:
                    # Analyze retrieval
                    analysis = retro_inference.analyze_retrieval(query)
                    print(f"Retrieved {analysis['num_chunks_retrieved']} chunks:")
                    for chunk_info in analysis['chunks']:
                        print(f"  {chunk_info['rank']}. Score: {chunk_info['similarity_score']:.3f}")
                        print(f"     Preview: {chunk_info['text_preview']}")
                        print()
                
                # Generate response
                response = retro_inference.generate_response(
                    query, 
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                
                print(f"RETRO Response: {response}")
                
        elif args.query:
            # Single query mode
            print(f"\nProcessing query: '{args.query}'")
            print("-" * 50)
            
            if args.analyze:
                # Analyze retrieval
                analysis = retro_inference.analyze_retrieval(args.query)
                print(f"Retrieved {analysis['num_chunks_retrieved']} chunks:")
                for chunk_info in analysis['chunks']:
                    print(f"  {chunk_info['rank']}. Score: {chunk_info['similarity_score']:.3f}")
                    print(f"     Preview: {chunk_info['text_preview']}")
                    print()
            
            # Generate response
            response = retro_inference.generate_response(
                args.query,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k
            )
            
            print(f"RETRO Response: {response}")
            
        else:
            print("Please specify --interactive or --query")
            
    except Exception as e:
        print(f"Inference failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
