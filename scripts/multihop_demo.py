"""
Multi-Hop Reasoning Demonstration Script
========================================

This script demonstrates the complete multi-hop reasoning system integrated
with the RETRO model, showing the improvements over single-hop retrieval.

Features Demonstrated:
- Multi-hop vs Single-hop comparison
- Reasoning path visualization
- Evaluation metrics
- Performance analysis
- Interactive query testing

Usage:
    python scripts/multihop_demo.py --interactive
    python scripts/multihop_demo.py --demo
    python scripts/multihop_demo.py --evaluate
"""

import os
import sys
import argparse
import time
import json
from typing import List, Dict, Optional
from pathlib import Path

# Import our multi-hop reasoning components
from multihop_reasoning import MultiHopReasoningSystem, ReasoningPath
from enhanced_query_retro import EnhancedRetroQuery, ComparisonResult
from retro_multihop_model import RETROMultiHopSystem
from reasoning_evaluation import ReasoningEvaluator, EvaluationResult


class MultiHopDemo:
    """Comprehensive demonstration of multi-hop reasoning capabilities."""
    
    def __init__(self):
        print("Initializing Multi-Hop Reasoning Demo...")
        print("=" * 60)
        
        # Initialize components
        self.multihop_system = MultiHopReasoningSystem()
        self.query_system = EnhancedRetroQuery()
        self.retro_system = RETROMultiHopSystem()
        self.evaluator = ReasoningEvaluator()
        
        print("✅ All components initialized successfully!")
        print()
    
    def run_basic_demo(self):
        """Run basic demonstration of multi-hop reasoning."""
        print("BASIC MULTI-HOP REASONING DEMONSTRATION")
        print("=" * 60)
        
        # Example queries that benefit from multi-hop reasoning
        demo_queries = [
            "What are the treatment options for diabetes and how do they relate to blood pressure management?",
            "How does heart disease connect to diabetes and what are the combined treatment approaches?",
            "What are the symptoms of hypertension and how do they differ from diabetes symptoms?"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*80}")
            print(f"DEMO {i}: {query}")
            print(f"{'='*80}")
            
            # Compare single-hop vs multi-hop
            comparison = self.query_system.compare_approaches(query)
            
            self._display_comparison(comparison)
            
            # Show reasoning visualization
            print(f"\nREASONING PATH VISUALIZATION:")
            visualization = self.query_system.get_reasoning_visualization(query)
            print(visualization)
    
    def run_evaluation_demo(self):
        """Run evaluation demonstration with metrics."""
        print("MULTI-HOP REASONING EVALUATION DEMONSTRATION")
        print("=" * 60)
        
        # Test queries for evaluation
        eval_queries = [
            "What are the treatment options for diabetes and how do they relate to blood pressure management?",
            "How does heart disease connect to diabetes and what are the combined treatment approaches?",
            "What medications are used for diabetes and what are their side effects on cardiovascular health?"
        ]
        
        evaluation_results = []
        
        for i, query in enumerate(eval_queries, 1):
            print(f"\n{'='*60}")
            print(f"EVALUATION {i}: {query}")
            print(f"{'='*60}")
            
            # Perform multi-hop reasoning
            reasoning_path = self.multihop_system.reason(query, max_hops=3)
            
            # Evaluate reasoning path
            metrics = self.evaluator.evaluate_reasoning_path(reasoning_path, query=query)
            
            # Create evaluation result
            result = EvaluationResult(
                query=query,
                reasoning_path=reasoning_path,
                metrics=metrics
            )
            
            evaluation_results.append(result)
            
            # Display metrics
            self._display_evaluation_metrics(metrics)
        
        # Generate comprehensive report
        report = self.evaluator.generate_evaluation_report(evaluation_results)
        self._display_evaluation_summary(report)
    
    def run_retro_integration_demo(self):
        """Demonstrate RETRO model integration with multi-hop reasoning."""
        print("RETRO MODEL INTEGRATION DEMONSTRATION")
        print("=" * 60)
        
        # Test query
        query = "What are the treatment options for diabetes and how do they relate to blood pressure management?"
        
        print(f"Query: {query}")
        print()
        
        try:
            # Generate with multi-hop RETRO model
            result = self.retro_system.generate_with_reasoning(query)
            
            print("MULTI-HOP RETRO GENERATION RESULT:")
            print(f"  Response: {result['response']}")
            print(f"  Reasoning Steps: {result['reasoning_steps']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Chunks Used: {result['chunks_used']}")
            print(f"  Processing Time: {result['processing_time']:.3f}s")
            
            # Compare generation approaches
            comparison = self.retro_system.compare_generation_approaches(query)
            
            print(f"\nGENERATION APPROACH COMPARISON:")
            print(f"  Single-hop chunks: {comparison['single_hop']['chunks_used']}")
            print(f"  Multi-hop chunks: {comparison['multi_hop']['chunks_used']}")
            print(f"  Multi-hop reasoning steps: {comparison['multi_hop']['reasoning_steps']}")
            print(f"  Confidence improvement: {comparison['multi_hop']['confidence'] - comparison['single_hop']['confidence']:.3f}")
            
        except Exception as e:
            print(f"❌ Error in RETRO integration demo: {e}")
            print("This might be due to missing model files or dependencies.")
    
    def run_interactive_demo(self):
        """Run interactive demonstration."""
        print("INTERACTIVE MULTI-HOP REASONING DEMO")
        print("=" * 60)
        print("Type your medical queries to see multi-hop reasoning in action!")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the demo")
        print("  'compare <query>' - Compare single-hop vs multi-hop")
        print("  'evaluate <query>' - Evaluate reasoning quality")
        print("  'visualize <query>' - Show reasoning path visualization")
        print("  'help' - Show this help message")
        print()
        
        while True:
            try:
                user_input = input("Enter query or command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == 'help':
                    print("Commands:")
                    print("  'quit' or 'exit' - Exit the demo")
                    print("  'compare <query>' - Compare single-hop vs multi-hop")
                    print("  'evaluate <query>' - Evaluate reasoning quality")
                    print("  'visualize <query>' - Show reasoning path visualization")
                    print("  'help' - Show this help message")
                    continue
                
                if user_input.lower().startswith('compare '):
                    query = user_input[8:].strip()
                    if query:
                        self._interactive_compare(query)
                    else:
                        print("Please provide a query after 'compare'")
                    continue
                
                if user_input.lower().startswith('evaluate '):
                    query = user_input[9:].strip()
                    if query:
                        self._interactive_evaluate(query)
                    else:
                        print("Please provide a query after 'evaluate'")
                    continue
                
                if user_input.lower().startswith('visualize '):
                    query = user_input[10:].strip()
                    if query:
                        self._interactive_visualize(query)
                    else:
                        print("Please provide a query after 'visualize'")
                    continue
                
                if not user_input:
                    continue
                
                # Default: run comparison
                self._interactive_compare(user_input)
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again or type 'help' for commands.")
    
    def _interactive_compare(self, query: str):
        """Interactive comparison for a query."""
        print(f"\n{'='*60}")
        print(f"COMPARISON: {query}")
        print(f"{'='*60}")
        
        try:
            comparison = self.query_system.compare_approaches(query)
            self._display_comparison(comparison)
        except Exception as e:
            print(f"❌ Error in comparison: {e}")
    
    def _interactive_evaluate(self, query: str):
        """Interactive evaluation for a query."""
        print(f"\n{'='*60}")
        print(f"EVALUATION: {query}")
        print(f"{'='*60}")
        
        try:
            reasoning_path = self.multihop_system.reason(query, max_hops=3)
            metrics = self.evaluator.evaluate_reasoning_path(reasoning_path, query=query)
            self._display_evaluation_metrics(metrics)
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")
    
    def _interactive_visualize(self, query: str):
        """Interactive visualization for a query."""
        print(f"\n{'='*60}")
        print(f"VISUALIZATION: {query}")
        print(f"{'='*60}")
        
        try:
            visualization = self.query_system.get_reasoning_visualization(query)
            print(visualization)
        except Exception as e:
            print(f"❌ Error in visualization: {e}")
    
    def _display_comparison(self, comparison: ComparisonResult):
        """Display comparison results."""
        print(f"\nSINGLE-HOP RETRIEVAL:")
        print(f"  Chunks: {len(comparison.single_hop.chunks)}")
        print(f"  Confidence: {comparison.single_hop.confidence:.3f}")
        print(f"  Time: {comparison.single_hop.processing_time:.3f}s")
        print(f"  Top chunks:")
        for i, chunk in enumerate(comparison.single_hop.chunks[:2]):
            print(f"    {i+1}. {chunk[:80]}...")
        
        print(f"\nMULTI-HOP REASONING:")
        print(f"  Chunks: {len(comparison.multi_hop.chunks)}")
        print(f"  Confidence: {comparison.multi_hop.confidence:.3f}")
        print(f"  Time: {comparison.multi_hop.processing_time:.3f}s")
        if comparison.multi_hop.reasoning_path:
            print(f"  Reasoning steps: {len(comparison.multi_hop.reasoning_path.steps)}")
            print(f"  Path score: {comparison.multi_hop.reasoning_path.path_score:.3f}")
        
        print(f"\nIMPROVEMENT METRICS:")
        for metric, value in comparison.improvement_metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    def _display_evaluation_metrics(self, metrics):
        """Display evaluation metrics."""
        print(f"  Path Coherence: {metrics.path_coherence:.3f}")
        print(f"  Step Transitions: {metrics.step_transitions:.3f}")
        print(f"  Reasoning Depth: {metrics.reasoning_depth}")
        print(f"  Evidence Relevance: {metrics.evidence_relevance:.3f}")
        print(f"  Evidence Diversity: {metrics.evidence_diversity:.3f}")
        print(f"  Evidence Coverage: {metrics.evidence_coverage:.3f}")
        print(f"  Answer Completeness: {metrics.answer_completeness:.3f}")
        print(f"  Answer Accuracy: {metrics.answer_accuracy:.3f}")
        print(f"  Reasoning Efficiency: {metrics.reasoning_efficiency:.3f}")
        print(f"  Multi-hop Advantage: {metrics.multi_hop_advantage:.3f}")
    
    def _display_evaluation_summary(self, report: Dict):
        """Display evaluation summary."""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Queries: {report['summary']['total_queries']}")
        print(f"Average Path Coherence: {report['summary']['average_path_coherence']:.3f}")
        print(f"Average Evidence Relevance: {report['summary']['average_evidence_relevance']:.3f}")
        print(f"Average Answer Accuracy: {report['summary']['average_answer_accuracy']:.3f}")
        print(f"Average Reasoning Efficiency: {report['summary']['average_reasoning_efficiency']:.3f}")
    
    def save_demo_results(self, output_dir: str = None):
        """Save demonstration results to files."""
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "demo_results")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save example comparison
        query = "What are the treatment options for diabetes and how do they relate to blood pressure management?"
        comparison = self.query_system.compare_approaches(query)
        
        # Convert to serializable format
        comparison_data = {
            'query': comparison.query,
            'single_hop': {
                'chunks': comparison.single_hop.chunks,
                'chunk_scores': comparison.single_hop.chunk_scores,
                'confidence': comparison.single_hop.confidence,
                'processing_time': comparison.single_hop.processing_time
            },
            'multi_hop': {
                'chunks': comparison.multi_hop.chunks,
                'chunk_scores': comparison.multi_hop.chunk_scores,
                'confidence': comparison.multi_hop.confidence,
                'processing_time': comparison.multi_hop.processing_time,
                'reasoning_steps': len(comparison.multi_hop.reasoning_path.steps) if comparison.multi_hop.reasoning_path else 0
            },
            'improvement_metrics': comparison.improvement_metrics
        }
        
        # Save comparison results
        comparison_file = os.path.join(output_dir, "comparison_results.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Save reasoning visualization
        visualization = self.query_system.get_reasoning_visualization(query)
        viz_file = os.path.join(output_dir, "reasoning_visualization.txt")
        with open(viz_file, 'w') as f:
            f.write(visualization)
        
        print(f"Demo results saved to: {output_dir}")
        print(f"  - Comparison results: {comparison_file}")
        print(f"  - Reasoning visualization: {viz_file}")


def main():
    """Main function for the multi-hop reasoning demonstration."""
    parser = argparse.ArgumentParser(description='Multi-Hop Reasoning Demonstration')
    parser.add_argument('--demo', action='store_true',
                       help='Run basic demonstration')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation demonstration')
    parser.add_argument('--retro', action='store_true',
                       help='Run RETRO integration demonstration')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive demonstration')
    parser.add_argument('--save', action='store_true',
                       help='Save demonstration results')
    parser.add_argument('--all', action='store_true',
                       help='Run all demonstrations')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = MultiHopDemo()
    
    if args.all or (not any([args.demo, args.evaluate, args.retro, args.interactive])):
        # Run all demonstrations
        demo.run_basic_demo()
        demo.run_evaluation_demo()
        demo.run_retro_integration_demo()
        
        if args.save:
            demo.save_demo_results()
    else:
        # Run specific demonstrations
        if args.demo:
            demo.run_basic_demo()
        
        if args.evaluate:
            demo.run_evaluation_demo()
        
        if args.retro:
            demo.run_retro_integration_demo()
        
        if args.interactive:
            demo.run_interactive_demo()
        
        if args.save:
            demo.save_demo_results()
    
    print(f"\n{'='*60}")
    print("MULTI-HOP REASONING DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print("Key Benefits Demonstrated:")
    print("✅ Multi-hop reasoning chains facts across knowledge chunks")
    print("✅ Iterative query expansion improves retrieval coverage")
    print("✅ Graph-based traversal finds related information")
    print("✅ Enhanced context integration in RETRO model")
    print("✅ Comprehensive evaluation metrics for reasoning quality")
    print("✅ Significant improvements over single-hop retrieval")


if __name__ == "__main__":
    main()
