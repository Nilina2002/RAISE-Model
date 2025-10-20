import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple

from retro_model import RETROModel, RETRORetrievalSystem, create_retro_model


class RETRODataset(Dataset):
    """Dataset class for RETRO model training."""
    
    def __init__(self, texts: List[str], retrieval_system: RETRORetrievalSystem,
                 max_length: int = 512, chunk_size: int = 64, num_retrieved_chunks: int = 5):
        self.texts = texts
        self.retrieval_system = retrieval_system
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.num_retrieved_chunks = num_retrieved_chunks
        
        # Simple vocabulary (in practice, use proper tokenizer)
        self.vocab = {}
        self.id_to_token = {}
        self._build_vocab()
        
    def _build_vocab(self):
        """Build vocabulary from all texts."""
        all_tokens = set()
        for text in self.texts:
            tokens = text.lower().split()
            all_tokens.update(tokens)
            
        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        all_tokens = special_tokens + sorted(list(all_tokens))
        
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
    def tokenize(self, text: str) -> List[int]:
        """Simple tokenization."""
        tokens = text.lower().split()
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return ' '.join([self.id_to_token.get(tid, '<unk>') for tid in token_ids])
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        tokens = self.tokenize(text)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            
        # Create input and target (shifted by 1 for language modeling)
        input_ids = tokens[:-1] if len(tokens) > 1 else tokens
        target_ids = tokens[1:] if len(tokens) > 1 else tokens
        
        # Pad sequences
        input_ids = input_ids + [self.vocab['<pad>']] * (self.max_length - len(input_ids))
        target_ids = target_ids + [self.vocab['<pad>']] * (self.max_length - len(target_ids))
        
        # Retrieve chunks for the first part of the sequence
        query = text[:200]  # Use first 200 characters as query
        retrieved_chunks = self.retrieval_system.retrieve_chunks(query, self.num_retrieved_chunks)
        
        # Encode chunks
        chunk_embeddings = self.retrieval_system.encode_chunks(retrieved_chunks)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'chunk_embeddings': chunk_embeddings,
            'attention_mask': torch.tensor([1] * len(tokens[:-1]) + [0] * (self.max_length - len(tokens[:-1])), dtype=torch.long)
        }


class RETROTrainer:
    """Trainer class for RETRO model."""
    
    def __init__(self, model: RETROModel, dataset: RETRODataset, device: str = 'cpu'):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab['<pad>'])
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            chunk_embeddings = batch['chunk_embeddings'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, chunk_embeddings, attention_mask)
            
            # Calculate loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
        
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                chunk_embeddings = batch['chunk_embeddings'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, chunk_embeddings, attention_mask)
                
                # Calculate loss
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches


def prepare_training_data(corpus_path: str, max_samples: int = 1000) -> List[str]:
    """Prepare training data from the corpus."""
    with open(corpus_path, "rb") as f:
        full_corpus = pickle.load(f)
    
    # Filter and limit samples
    filtered_texts = []
    for text in full_corpus:
        if len(text.strip()) > 50:  # Only use texts with reasonable length
            filtered_texts.append(text.strip())
            if len(filtered_texts) >= max_samples:
                break
                
    return filtered_texts


def train_retro_model(corpus_path: str, output_dir: str, 
                     num_epochs: int = 10, batch_size: int = 8, learning_rate: float = 1e-4,
                     max_samples: int = 1000, device: str = 'cpu'):
    """Main training function."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing training data...")
    texts = prepare_training_data(corpus_path, max_samples)
    print(f"Prepared {len(texts)} training samples")
    
    # Initialize retrieval system
    print("Initializing retrieval system...")
    retrieval_system = RETRORetrievalSystem()
    
    # Create dataset
    print("Creating dataset...")
    dataset = RETRODataset(texts, retrieval_system)
    print(f"Vocabulary size: {len(dataset.vocab)}")
    
    # Create model
    print("Creating RETRO model...")
    model = create_retro_model(vocab_size=len(dataset.vocab))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    trainer = RETROTrainer(model, dataset, device)
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab': dataset.vocab,
                'vocab_size': len(dataset.vocab),
                'd_model': model.d_model,
                'n_heads': model.attention.n_heads if hasattr(model, 'attention') else 8,
                'n_encoder_layers': len(model.encoder_layers),
                'n_decoder_layers': len(model.decoder_layers),
                'd_ff': model.decoder_layers[0].feed_forward[0].out_features if model.decoder_layers else 2048,
                'chunk_size': model.chunk_size,
                'window_size': 32  # Default value
            }
            
            model_path = os.path.join(output_dir, 'best_retro_model.pth')
            torch.save(checkpoint, model_path)
            print(f"Best model saved to {model_path}")
            
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'vocab_size': len(dataset.vocab)
        }
        
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
    print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
    return model, dataset.vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RETRO model')
    parser.add_argument('--corpus_path', type=str, 
                       default='../data/processed_chunks/full_corpus.pkl',
                       help='Path to the corpus pickle file')
    parser.add_argument('--output_dir', type=str, default='../models/retro',
                       help='Output directory for trained model')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of training samples')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Adjust paths for script execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    corpus_path = os.path.join(project_root, "data", "processed_chunks", "full_corpus.pkl")
    output_dir = os.path.join(project_root, "models", "retro")
    
    print("RETRO Model Training")
    print("===================")
    print(f"Corpus path: {corpus_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    
    try:
        model, vocab = train_retro_model(
            corpus_path=corpus_path,
            output_dir=output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
            device=args.device
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
