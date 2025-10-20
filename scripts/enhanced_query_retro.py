"""
Enhanced Query System with Multi-Hop Reasoning
==============================================

This module extends the original query_retro.py with multi-hop reasoning capabilities,
enabling the system to chain facts and perform iterative reasoning across multiple
knowledge chunks.

Key Enhancements:
- Multi-hop reasoning with graph traversal
- Iterative query expansion and refinement
- Reasoning path tracking and validation
- Integration with existing RETRO architecture
- Comparison between single-hop and multi-hop approaches

Usage:
    from scripts.enhanced_query_retro import EnhancedRetroQuery
    
    query_system = EnhancedRetroQuery()
    
    # Single-hop retrieval (original behavior)
    results = query_system.retrieve(query, top_k=5)
    
    # Multi-hop reasoning
    reasoning_result = query_system.reason(query, max_hops=3)
    
    # Comparison
    comparison = query_system.compare_approaches(query)
"""

import pickle
import faiss
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import networkx as nx
from collections import defaultdict
import re
import time

# Import our multi-hop reasoning components
from multihop_reasoning import (
    MultiHopReasoningSystem, 
    ReasoningPath, 
    ReasoningStep,
    KnowledgeGraph,
    QueryExpander,
    ReasoningEngine
)


@dataclass
class QueryResult:
    """Enhanced query result with reasoning information."""
    query: str
    chunks: List[str]
    chunk_scores: List[float]
    reasoning_path: Optional[ReasoningPath] = None
    approach: str = "single_hop"  # "single_hop" or "multi_hop"
    processing_time: float = 0.0
    confidence: float = 0.0


@dataclass
class ComparisonResult:
    """Result comparing single-hop vs multi-hop approaches."""
    query: str
    single_hop: QueryResult
    multi_hop: QueryResult
    improvement_metrics: Dict[str, float]


class EnhancedRetroQuery:
    """Enhanced RETRO query system with multi-hop reasoning capabilities."""
    
    def __init__(self, corpus_path: str = None, index_path: str = None, 
                 embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Get the directory of this script and construct the paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        self.corpus_path = corpus_path or os.path.join(project_root, "data", "processed_chunks", "full_corpus.pkl")
        self.index_path = index_path or os.path.join(project_root, "faiss_index", "medical_faiss.index")
        
        # Load components
        self._load_components(embedder_model)
        
        # Initialize multi-hop reasoning system
        self.multihop_system = MultiHopReasoningSystem(
            self.corpus_path, self.index_path, embedder_model
        )
        
        print("Enhanced RETRO Query System initialized with multi-hop reasoning capabilities")
    
    def _load_components(self, embedder_model: str):
        """Load the basic retrieval system components."""
        print(f"Loading enhanced retrieval system with model: {embedder_model}")
        
        # Load corpus
        with open(self.corpus_path, "rb") as f:
            self.full_corpus = pickle.load(f)
            
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load embedder
        self.embedder = SentenceTransformer(embedder_model)
        
        print(f"Enhanced retrieval system loaded. Corpus size: {len(self.full_corpus)}")
    
    def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        """
        Original single-hop retrieval method (backward compatible).
        
        Args:
            query: The search query
            top_k: Number of top chunks to retrieve
            
        Returns:
            QueryResult with retrieved chunks and scores
        """
        start_time = time.time()
        
        # Perform retrieval
        query_emb = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_emb, top_k)
        
        # Convert distances to similarity scores (higher is better)
        scores = 1.0 / (1.0 + distances[0])  # Convert distance to similarity
        
        # Get chunks
        chunks = [self.full_corpus[i] for i in indices[0]]
        
        processing_time = time.time() - start_time
        
        return QueryResult(
            query=query,
            chunks=chunks,
            chunk_scores=scores.tolist(),
            approach="single_hop",
            processing_time=processing_time,
            confidence=np.mean(scores)
        )
    
    def reason(self, query: str, max_hops: int = 3, top_k: int = 10) -> QueryResult:
        """
        Multi-hop reasoning method.
        
        Args:
            query: The search query
            max_hops: Maximum number of reasoning hops
            top_k: Initial number of chunks to retrieve
            
        Returns:
            QueryResult with reasoning path and enhanced information
        """
        start_time = time.time()
        
        # Perform multi-hop reasoning
        reasoning_path = self.multihop_system.reason(query, max_hops, top_k)
        
        # Extract all chunks from reasoning steps
        all_chunks = []
        all_scores = []
        
        for step in reasoning_path.steps:
            for chunk in step.retrieved_chunks:
                if chunk not in all_chunks:
                    all_chunks.append(chunk)
                    all_scores.append(step.confidence)
        
        processing_time = time.time() - start_time
        
        return QueryResult(
            query=query,
            chunks=all_chunks,
            chunk_scores=all_scores,
            reasoning_path=reasoning_path,
            approach="multi_hop",
            processing_time=processing_time,
            confidence=reasoning_path.confidence
        )
    
    def compare_approaches(self, query: str, top_k: int = 5, max_hops: int = 3) -> ComparisonResult:
        """
        Compare single-hop vs multi-hop approaches for a given query.
        
        Args:
            query: The search query
            top_k: Number of chunks for single-hop
            max_hops: Maximum hops for multi-hop
            
        Returns:
            ComparisonResult with both approaches and improvement metrics
        """
        print(f"Comparing approaches for query: '{query}'")
        
        # Single-hop retrieval
        single_hop_result = self.retrieve(query, top_k)
        
        # Multi-hop reasoning
        multi_hop_result = self.reason(query, max_hops, top_k)
        
        # Calculate improvement metrics
        improvement_metrics = self._calculate_improvement_metrics(
            single_hop_result, multi_hop_result
        )
        
        return ComparisonResult(
            query=query,
            single_hop=single_hop_result,
            multi_hop=multi_hop_result,
            improvement_metrics=improvement_metrics
        )
    
    def _calculate_improvement_metrics(self, single_hop: QueryResult, 
                                     multi_hop: QueryResult) -> Dict[str, float]:
        """Calculate metrics showing improvement from multi-hop reasoning."""
        metrics = {}
        
        # Coverage improvement (number of unique chunks)
        single_hop_coverage = len(single_hop.chunks)
        multi_hop_coverage = len(multi_hop.chunks)
        metrics['coverage_improvement'] = (multi_hop_coverage - single_hop_coverage) / single_hop_coverage
        
        # Confidence improvement
        metrics['confidence_improvement'] = multi_hop.confidence - single_hop.confidence
        
        # Reasoning depth (number of reasoning steps)
        if multi_hop.reasoning_path:
            metrics['reasoning_depth'] = len(multi_hop.reasoning_path.steps)
        else:
            metrics['reasoning_depth'] = 0
        
        # Evidence quality (number of evidence pieces)
        if multi_hop.reasoning_path:
            total_evidence = sum(len(step.evidence) for step in multi_hop.reasoning_path.steps)
            metrics['evidence_quality'] = total_evidence
        else:
            metrics['evidence_quality'] = 0
        
        # Processing time ratio
        metrics['time_ratio'] = multi_hop.processing_time / single_hop.processing_time if single_hop.processing_time > 0 else 1.0
        
        return metrics
    
    def analyze_reasoning_path(self, query: str, max_hops: int = 3) -> Dict:
        """Analyze the reasoning path in detail."""
        reasoning_path = self.multihop_system.reason(query, max_hops)
        
        analysis = {
            'query': query,
            'total_steps': len(reasoning_path.steps),
            'final_confidence': reasoning_path.confidence,
            'path_score': reasoning_path.path_score,
            'steps': []
        }
        
        for step in reasoning_path.steps:
            step_analysis = {
                'step_id': step.step_id,
                'query': step.query,
                'num_chunks': len(step.retrieved_chunks),
                'confidence': step.confidence,
                'reasoning': step.reasoning,
                'evidence_count': len(step.evidence),
                'evidence_preview': step.evidence[:2] if step.evidence else []
            }
            analysis['steps'].append(step_analysis)
        
        analysis['final_answer'] = reasoning_path.final_answer
        
        return analysis
    
    def get_reasoning_visualization(self, query: str, max_hops: int = 3) -> str:
        """Generate a text visualization of the reasoning path."""
        reasoning_path = self.multihop_system.reason(query, max_hops)
        
        visualization = f"REASONING PATH VISUALIZATION\n"
        visualization += f"Query: {query}\n"
        visualization += f"{'='*60}\n\n"
        
        for i, step in enumerate(reasoning_path.steps):
            visualization += f"STEP {step.step_id}:\n"
            visualization += f"  Query: {step.query}\n"
            visualization += f"  Chunks Retrieved: {len(step.retrieved_chunks)}\n"
            visualization += f"  Confidence: {step.confidence:.3f}\n"
            visualization += f"  Reasoning: {step.reasoning[:100]}...\n"
            visualization += f"  Evidence: {len(step.evidence)} pieces\n"
            visualization += f"  {'-'*40}\n\n"
        
        visualization += f"FINAL ANSWER:\n"
        visualization += f"{reasoning_path.final_answer}\n\n"
        visualization += f"Overall Confidence: {reasoning_path.confidence:.3f}\n"
        visualization += f"Path Score: {reasoning_path.path_score:.3f}\n"
        
        return visualization
    
    def save_comparison_report(self, query: str, output_path: str = None) -> str:
        """Save a detailed comparison report to file."""
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, f"comparison_report_{int(time.time())}.json")
        
        comparison = self.compare_approaches(query)
        
        # Convert to serializable format
        report = {
            'query': comparison.query,
            'timestamp': time.time(),
            'single_hop': asdict(comparison.single_hop),
            'multi_hop': asdict(comparison.multi_hop),
            'improvement_metrics': comparison.improvement_metrics
        }
        
        # Handle non-serializable objects
        if report['multi_hop']['reasoning_path']:
            report['multi_hop']['reasoning_path'] = {
                'steps': [asdict(step) for step in comparison.multi_hop.reasoning_path.steps],
                'final_answer': comparison.multi_hop.reasoning_path.final_answer,
                'confidence': comparison.multi_hop.reasoning_path.confidence,
                'path_score': comparison.multi_hop.reasoning_path.path_score
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comparison report saved to: {output_path}")
        return output_path


def main():
    """Demonstration of enhanced query system."""
    print("Enhanced RETRO Query System Demo")
    print("=" * 50)
    
    # Initialize enhanced query system
    query_system = EnhancedRetroQuery()
    
    # Example queries that benefit from multi-hop reasoning
    test_queries = [
        "What are the treatment options for diabetes and how do they relate to blood pressure management?",
        "How does heart disease connect to diabetes and what are the combined treatment approaches?",
        "What are the symptoms of hypertension and how do they differ from diabetes symptoms?",
        "What medications are used for diabetes and what are their side effects on cardiovascular health?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print(f"{'='*80}")
        
        # Compare approaches
        comparison = query_system.compare_approaches(query)
        
        print(f"\nSINGLE-HOP RETRIEVAL:")
        print(f"  Chunks: {len(comparison.single_hop.chunks)}")
        print(f"  Confidence: {comparison.single_hop.confidence:.3f}")
        print(f"  Time: {comparison.single_hop.processing_time:.3f}s")
        print(f"  Top chunks:")
        for j, chunk in enumerate(comparison.single_hop.chunks[:2]):
            print(f"    {j+1}. {chunk[:80]}...")
        
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
        
        # Show reasoning visualization
        print(f"\nREASONING VISUALIZATION:")
        visualization = query_system.get_reasoning_visualization(query)
        print(visualization)
        
        # Save report for first query
        if i == 1:
            report_path = query_system.save_comparison_report(query)
            print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
