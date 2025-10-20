"""
Complete RETRO Model Workflow
============================

This script demonstrates the complete workflow from data preprocessing
to RETRO model training and inference. It serves as a comprehensive
example of how to use the RETRO model implementation.

Usage:
    python scripts/retro_workflow.py --demo          # Run demonstration
    python scripts/retro_workflow.py --train         # Train the model
    python scripts/retro_workflow.py --inference     # Run inference
    python scripts/retro_workflow.py --full          # Complete workflow
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("Checking Prerequisites...")
    print("=" * 40)
    
    # Check if data exists
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    corpus_path = project_root / "data" / "processed_chunks" / "full_corpus.pkl"
    index_path = project_root / "faiss_index" / "medical_faiss.index"
    
    if not corpus_path.exists():
        print("❌ Corpus not found. Please run data preprocessing first.")
        return False
    else:
        print("✅ Corpus found")
        
    if not index_path.exists():
        print("❌ FAISS index not found. Please build the index first.")
        return False
    else:
        print("✅ FAISS index found")
        
    # Check if required Python packages are installed
    try:
        import torch
        import transformers
        import sentence_transformers
        import faiss
        print("✅ Required packages installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False


def run_demonstration():
    """Run the RETRO model demonstration."""
    print("\n" + "="*60)
    print("RETRO MODEL DEMONSTRATION")
    print("="*60)
    
    script_dir = Path(__file__).parent
    demo_script = script_dir / "retro_example.py"
    
    if not demo_script.exists():
        print("❌ Demo script not found")
        return False
        
    return run_command(
        f"python {demo_script}",
        "Running RETRO Model Demonstration"
    )


def run_training():
    """Run RETRO model training."""
    print("\n" + "="*60)
    print("RETRO MODEL TRAINING")
    print("="*60)
    
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_retro.py"
    
    if not train_script.exists():
        print("❌ Training script not found")
        return False
        
    # Training parameters
    num_epochs = 5  # Reduced for demo
    batch_size = 4  # Reduced for demo
    max_samples = 500  # Reduced for demo
    
    return run_command(
        f"python {train_script} --num_epochs {num_epochs} --batch_size {batch_size} --max_samples {max_samples}",
        f"Training RETRO Model (epochs: {num_epochs}, batch_size: {batch_size})"
    )


def run_inference():
    """Run RETRO model inference."""
    print("\n" + "="*60)
    print("RETRO MODEL INFERENCE")
    print("="*60)
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_path = project_root / "models" / "retro" / "best_retro_model.pth"
    
    if not model_path.exists():
        print("❌ Trained model not found. Please train the model first.")
        return False
        
    inference_script = script_dir / "retro_inference.py"
    
    if not inference_script.exists():
        print("❌ Inference script not found")
        return False
        
    # Run inference with example queries
    example_queries = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What is coronary artery disease?"
    ]
    
    success = True
    for query in example_queries:
        success &= run_command(
            f'python {inference_script} --query "{query}" --analyze',
            f"Inference for: {query}"
        )
        
    return success


def run_full_workflow():
    """Run the complete RETRO model workflow."""
    print("\n" + "="*60)
    print("COMPLETE RETRO MODEL WORKFLOW")
    print("="*60)
    
    steps = [
        ("Prerequisites Check", check_prerequisites),
        ("Demonstration", run_demonstration),
        ("Model Training", run_training),
        ("Model Inference", run_inference)
    ]
    
    results = []
    for step_name, step_function in steps:
        print(f"\n{'='*60}")
        print(f"WORKFLOW STEP: {step_name}")
        print(f"{'='*60}")
        
        success = step_function()
        results.append((step_name, success))
        
        if not success:
            print(f"\n❌ Workflow stopped at step: {step_name}")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("WORKFLOW SUMMARY")
    print(f"{'='*60}")
    
    for step_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{step_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 Complete RETRO model workflow executed successfully!")
    else:
        print("\n⚠️  Some steps failed. Please check the errors above.")
    
    return all_passed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='RETRO Model Workflow')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration only')
    parser.add_argument('--train', action='store_true',
                       help='Run training only')
    parser.add_argument('--inference', action='store_true',
                       help='Run inference only')
    parser.add_argument('--full', action='store_true',
                       help='Run complete workflow')
    parser.add_argument('--check', action='store_true',
                       help='Check prerequisites only')
    
    args = parser.parse_args()
    
    if args.check:
        check_prerequisites()
    elif args.demo:
        run_demonstration()
    elif args.train:
        run_training()
    elif args.inference:
        run_inference()
    elif args.full:
        run_full_workflow()
    else:
        print("RETRO Model Workflow")
        print("===================")
        print("\nThis script provides a complete workflow for the RETRO model.")
        print("\nAvailable options:")
        print("  --check      Check prerequisites")
        print("  --demo       Run demonstration")
        print("  --train      Train the model")
        print("  --inference  Run inference")
        print("  --full       Complete workflow")
        print("\nExample usage:")
        print("  python scripts/retro_workflow.py --full")
        print("  python scripts/retro_workflow.py --demo")


if __name__ == "__main__":
    main()

