"""
Multi-Hop Reasoning Evaluation Metrics
======================================

This module provides comprehensive evaluation metrics for multi-hop reasoning
quality, including path validation, reasoning coherence, and answer accuracy.

Key Metrics:
- Reasoning Path Coherence
- Evidence Quality and Relevance
- Answer Completeness and Accuracy
- Multi-hop vs Single-hop Comparison
- Reasoning Efficiency Metrics

Usage:
    from scripts.reasoning_evaluation import ReasoningEvaluator
    
    evaluator = ReasoningEvaluator()
    metrics = evaluator.evaluate_reasoning_path(reasoning_path, ground_truth)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import json
import time
from collections import defaultdict
import re
from sentence_transformers import SentenceTransformer
import networkx as nx

# Import our multi-hop reasoning components
from multihop_reasoning import ReasoningPath, ReasoningStep, MultiHopReasoningSystem
from enhanced_query_retro import EnhancedRetroQuery, ComparisonResult


@dataclass
class ReasoningMetrics:
    """Comprehensive metrics for reasoning evaluation."""
    # Path Quality Metrics
    path_coherence: float
    step_transitions: float
    reasoning_depth: int
    
    # Evidence Quality Metrics
    evidence_relevance: float
    evidence_diversity: float
    evidence_coverage: float
    
    # Answer Quality Metrics
    answer_completeness: float
    answer_accuracy: float
    answer_confidence: float
    
    # Efficiency Metrics
    processing_time: float
    chunks_utilized: int
    reasoning_efficiency: float
    
    # Comparison Metrics
    improvement_over_single_hop: float
    multi_hop_advantage: float


@dataclass
class EvaluationResult:
    """Complete evaluation result with all metrics."""
    query: str
    reasoning_path: ReasoningPath
    metrics: ReasoningMetrics
    ground_truth: Optional[str] = None
    evaluation_time: float = 0.0


class ReasoningEvaluator:
    """Comprehensive evaluator for multi-hop reasoning systems."""
    
    def __init__(self, embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder_model)
        self.evaluation_cache = {}
        
    def evaluate_reasoning_path(self, reasoning_path: ReasoningPath, 
                              ground_truth: Optional[str] = None,
                              query: str = "") -> ReasoningMetrics:
        """Evaluate a complete reasoning path."""
        start_time = time.time()
        
        # Path Quality Metrics
        path_coherence = self._evaluate_path_coherence(reasoning_path)
        step_transitions = self._evaluate_step_transitions(reasoning_path)
        reasoning_depth = len(reasoning_path.steps)
        
        # Evidence Quality Metrics
        evidence_relevance = self._evaluate_evidence_relevance(reasoning_path, query)
        evidence_diversity = self._evaluate_evidence_diversity(reasoning_path)
        evidence_coverage = self._evaluate_evidence_coverage(reasoning_path)
        
        # Answer Quality Metrics
        answer_completeness = self._evaluate_answer_completeness(reasoning_path, ground_truth)
        answer_accuracy = self._evaluate_answer_accuracy(reasoning_path, ground_truth)
        answer_confidence = reasoning_path.confidence
        
        # Efficiency Metrics
        processing_time = self._estimate_processing_time(reasoning_path)
        chunks_utilized = sum(len(step.retrieved_chunks) for step in reasoning_path.steps)
        reasoning_efficiency = self._calculate_reasoning_efficiency(reasoning_path)
        
        # Comparison Metrics (placeholder - would need single-hop baseline)
        improvement_over_single_hop = 0.0  # Would be calculated with baseline
        multi_hop_advantage = self._calculate_multi_hop_advantage(reasoning_path)
        
        evaluation_time = time.time() - start_time
        
        return ReasoningMetrics(
            path_coherence=path_coherence,
            step_transitions=step_transitions,
            reasoning_depth=reasoning_depth,
            evidence_relevance=evidence_relevance,
            evidence_diversity=evidence_diversity,
            evidence_coverage=evidence_coverage,
            answer_completeness=answer_completeness,
            answer_accuracy=answer_accuracy,
            answer_confidence=answer_confidence,
            processing_time=processing_time,
            chunks_utilized=chunks_utilized,
            reasoning_efficiency=reasoning_efficiency,
            improvement_over_single_hop=improvement_over_single_hop,
            multi_hop_advantage=multi_hop_advantage
        )
    
    def _evaluate_path_coherence(self, reasoning_path: ReasoningPath) -> float:
        """Evaluate how coherent the reasoning path is."""
        if len(reasoning_path.steps) < 2:
            return 1.0  # Single step is trivially coherent
        
        coherence_scores = []
        
        for i in range(len(reasoning_path.steps) - 1):
            current_step = reasoning_path.steps[i]
            next_step = reasoning_path.steps[i + 1]
            
            # Calculate semantic similarity between consecutive reasoning steps
            current_reasoning = current_step.reasoning
            next_reasoning = next_step.reasoning
            
            # Encode reasoning texts
            current_emb = self.embedder.encode([current_reasoning], convert_to_numpy=True)
            next_emb = self.embedder.encode([next_reasoning], convert_to_numpy=True)
            
            # Calculate cosine similarity
            similarity = np.dot(current_emb[0], next_emb[0]) / (
                np.linalg.norm(current_emb[0]) * np.linalg.norm(next_emb[0])
            )
            
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _evaluate_step_transitions(self, reasoning_path: ReasoningPath) -> float:
        """Evaluate how well reasoning steps transition to each other."""
        if len(reasoning_path.steps) < 2:
            return 1.0
        
        transition_scores = []
        
        for i in range(len(reasoning_path.steps) - 1):
            current_step = reasoning_path.steps[i]
            next_step = reasoning_path.steps[i + 1]
            
            # Check if next step builds on current step
            current_entities = self._extract_entities(current_step.reasoning)
            next_entities = self._extract_entities(next_step.reasoning)
            
            # Calculate entity overlap
            if current_entities and next_entities:
                overlap = len(current_entities.intersection(next_entities)) / len(current_entities.union(next_entities))
                transition_scores.append(overlap)
            else:
                transition_scores.append(0.0)
        
        return np.mean(transition_scores) if transition_scores else 0.0
    
    def _evaluate_evidence_relevance(self, reasoning_path: ReasoningPath, query: str) -> float:
        """Evaluate how relevant the evidence is to the query."""
        if not query:
            return 0.0
        
        relevance_scores = []
        
        for step in reasoning_path.steps:
            for evidence in step.evidence:
                # Calculate similarity between query and evidence
                query_emb = self.embedder.encode([query], convert_to_numpy=True)
                evidence_emb = self.embedder.encode([evidence], convert_to_numpy=True)
                
                similarity = np.dot(query_emb[0], evidence_emb[0]) / (
                    np.linalg.norm(query_emb[0]) * np.linalg.norm(evidence_emb[0])
                )
                
                relevance_scores.append(similarity)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def _evaluate_evidence_diversity(self, reasoning_path: ReasoningPath) -> float:
        """Evaluate how diverse the evidence is across reasoning steps."""
        all_evidence = []
        for step in reasoning_path.steps:
            all_evidence.extend(step.evidence)
        
        if len(all_evidence) < 2:
            return 1.0  # Single piece of evidence is trivially diverse
        
        # Calculate pairwise similarities between evidence pieces
        evidence_embs = self.embedder.encode(all_evidence, convert_to_numpy=True)
        similarities = np.dot(evidence_embs, evidence_embs.T)
        
        # Calculate diversity as 1 - average similarity
        n = len(all_evidence)
        avg_similarity = (np.sum(similarities) - n) / (n * (n - 1))  # Exclude diagonal
        diversity = 1.0 - avg_similarity
        
        return max(0.0, diversity)
    
    def _evaluate_evidence_coverage(self, reasoning_path: ReasoningPath) -> float:
        """Evaluate how well the evidence covers the reasoning path."""
        total_evidence = sum(len(step.evidence) for step in reasoning_path.steps)
        total_steps = len(reasoning_path.steps)
        
        if total_steps == 0:
            return 0.0
        
        # Coverage is the ratio of evidence to steps
        coverage = min(total_evidence / total_steps, 1.0)
        return coverage
    
    def _evaluate_answer_completeness(self, reasoning_path: ReasoningPath, 
                                    ground_truth: Optional[str] = None) -> float:
        """Evaluate how complete the final answer is."""
        if not reasoning_path.final_answer:
            return 0.0
        
        # Simple completeness based on answer length and evidence usage
        answer_length = len(reasoning_path.final_answer.split())
        evidence_count = sum(len(step.evidence) for step in reasoning_path.steps)
        
        # Normalize completeness score
        completeness = min(answer_length / 50.0, 1.0) * min(evidence_count / 5.0, 1.0)
        
        return completeness
    
    def _evaluate_answer_accuracy(self, reasoning_path: ReasoningPath, 
                                ground_truth: Optional[str] = None) -> float:
        """Evaluate how accurate the final answer is."""
        if not ground_truth or not reasoning_path.final_answer:
            return reasoning_path.confidence  # Use model confidence as proxy
        
        # Calculate semantic similarity between answer and ground truth
        answer_emb = self.embedder.encode([reasoning_path.final_answer], convert_to_numpy=True)
        truth_emb = self.embedder.encode([ground_truth], convert_to_numpy=True)
        
        similarity = np.dot(answer_emb[0], truth_emb[0]) / (
            np.linalg.norm(answer_emb[0]) * np.linalg.norm(truth_emb[0])
        )
        
        return similarity
    
    def _estimate_processing_time(self, reasoning_path: ReasoningPath) -> float:
        """Estimate processing time based on reasoning complexity."""
        # Simple estimation based on number of steps and chunks
        total_chunks = sum(len(step.retrieved_chunks) for step in reasoning_path.steps)
        estimated_time = len(reasoning_path.steps) * 0.1 + total_chunks * 0.01
        return estimated_time
    
    def _calculate_reasoning_efficiency(self, reasoning_path: ReasoningPath) -> float:
        """Calculate reasoning efficiency (quality per unit of effort)."""
        if reasoning_path.path_score == 0:
            return 0.0
        
        total_chunks = sum(len(step.retrieved_chunks) for step in reasoning_path.steps)
        if total_chunks == 0:
            return 0.0
        
        # Efficiency = path_score / chunks_used
        efficiency = reasoning_path.path_score / total_chunks
        return efficiency
    
    def _calculate_multi_hop_advantage(self, reasoning_path: ReasoningPath) -> float:
        """Calculate the advantage of multi-hop reasoning."""
        if len(reasoning_path.steps) <= 1:
            return 0.0
        
        # Advantage based on reasoning depth and confidence
        depth_bonus = min(len(reasoning_path.steps) / 3.0, 1.0)
        confidence_bonus = reasoning_path.confidence
        
        advantage = (depth_bonus + confidence_bonus) / 2.0
        return advantage
    
    def _extract_entities(self, text: str) -> set:
        """Extract entities from text (simplified version)."""
        # Simple entity extraction - in practice, use NER models
        medical_terms = [
            'diabetes', 'hypertension', 'cancer', 'heart', 'lung', 'brain',
            'blood', 'pressure', 'sugar', 'insulin', 'treatment', 'therapy',
            'surgery', 'medication', 'drug', 'disease', 'symptom', 'diagnosis'
        ]
        
        entities = set()
        text_lower = text.lower()
        
        for term in medical_terms:
            if term in text_lower:
                entities.add(term)
        
        return entities
    
    def compare_reasoning_approaches(self, query: str, 
                                   single_hop_result: Dict,
                                   multi_hop_result: Dict) -> Dict:
        """Compare single-hop vs multi-hop reasoning approaches."""
        comparison = {
            'query': query,
            'single_hop': {
                'chunks_used': single_hop_result.get('chunks_used', 0),
                'confidence': single_hop_result.get('confidence', 0.0),
                'processing_time': single_hop_result.get('processing_time', 0.0)
            },
            'multi_hop': {
                'chunks_used': multi_hop_result.get('chunks_used', 0),
                'confidence': multi_hop_result.get('confidence', 0.0),
                'processing_time': multi_hop_result.get('processing_time', 0.0),
                'reasoning_steps': multi_hop_result.get('reasoning_steps', 0)
            },
            'improvements': {}
        }
        
        # Calculate improvements
        if comparison['single_hop']['chunks_used'] > 0:
            comparison['improvements']['chunk_coverage'] = (
                comparison['multi_hop']['chunks_used'] - comparison['single_hop']['chunks_used']
            ) / comparison['single_hop']['chunks_used']
        
        comparison['improvements']['confidence_gain'] = (
            comparison['multi_hop']['confidence'] - comparison['single_hop']['confidence']
        )
        
        if comparison['single_hop']['processing_time'] > 0:
            comparison['improvements']['time_ratio'] = (
                comparison['multi_hop']['processing_time'] / comparison['single_hop']['processing_time']
            )
        
        return comparison
    
    def generate_evaluation_report(self, evaluation_results: List[EvaluationResult]) -> Dict:
        """Generate a comprehensive evaluation report."""
        if not evaluation_results:
            return {}
        
        # Aggregate metrics
        all_metrics = [result.metrics for result in evaluation_results]
        
        report = {
            'summary': {
                'total_queries': len(evaluation_results),
                'average_path_coherence': np.mean([m.path_coherence for m in all_metrics]),
                'average_evidence_relevance': np.mean([m.evidence_relevance for m in all_metrics]),
                'average_answer_accuracy': np.mean([m.answer_accuracy for m in all_metrics]),
                'average_reasoning_efficiency': np.mean([m.reasoning_efficiency for m in all_metrics])
            },
            'detailed_results': []
        }
        
        for result in evaluation_results:
            detailed_result = {
                'query': result.query,
                'metrics': {
                    'path_coherence': result.metrics.path_coherence,
                    'evidence_relevance': result.metrics.evidence_relevance,
                    'answer_accuracy': result.metrics.answer_accuracy,
                    'reasoning_efficiency': result.metrics.reasoning_efficiency,
                    'multi_hop_advantage': result.metrics.multi_hop_advantage
                },
                'reasoning_depth': result.metrics.reasoning_depth,
                'chunks_utilized': result.metrics.chunks_utilized
            }
            report['detailed_results'].append(detailed_result)
        
        return report


def main():
    """Demonstration of reasoning evaluation system."""
    print("Multi-Hop Reasoning Evaluation Demo")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ReasoningEvaluator()
    
    # Initialize multi-hop system
    multihop_system = MultiHopReasoningSystem()
    
    # Test queries
    test_queries = [
        "What are the treatment options for diabetes and how do they relate to blood pressure management?",
        "How does heart disease connect to diabetes and what are the combined treatment approaches?"
    ]
    
    evaluation_results = []
    
    for query in test_queries:
        print(f"\nEvaluating query: {query}")
        
        # Perform multi-hop reasoning
        reasoning_path = multihop_system.reason(query, max_hops=3)
        
        # Evaluate reasoning path
        metrics = evaluator.evaluate_reasoning_path(reasoning_path, query=query)
        
        # Create evaluation result
        result = EvaluationResult(
            query=query,
            reasoning_path=reasoning_path,
            metrics=metrics
        )
        
        evaluation_results.append(result)
        
        # Print metrics
        print(f"  Path Coherence: {metrics.path_coherence:.3f}")
        print(f"  Evidence Relevance: {metrics.evidence_relevance:.3f}")
        print(f"  Answer Accuracy: {metrics.answer_accuracy:.3f}")
        print(f"  Reasoning Efficiency: {metrics.reasoning_efficiency:.3f}")
        print(f"  Multi-hop Advantage: {metrics.multi_hop_advantage:.3f}")
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(evaluation_results)
    
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total Queries: {report['summary']['total_queries']}")
    print(f"Average Path Coherence: {report['summary']['average_path_coherence']:.3f}")
    print(f"Average Evidence Relevance: {report['summary']['average_evidence_relevance']:.3f}")
    print(f"Average Answer Accuracy: {report['summary']['average_answer_accuracy']:.3f}")
    print(f"Average Reasoning Efficiency: {report['summary']['average_reasoning_efficiency']:.3f}")


if __name__ == "__main__":
    main()
