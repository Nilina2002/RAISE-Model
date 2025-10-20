"""
Multi-Hop Reasoning System for RETRO Model
==========================================

This module implements multi-hop reasoning capabilities for the RETRO model,
enabling the system to chain facts and perform iterative reasoning across
multiple knowledge chunks.

Key Features:
- Graph-based knowledge representation
- Iterative query expansion and refinement
- Multi-hop retrieval with reasoning paths
- Integration with existing RETRO architecture
- Reasoning path validation and scoring

Usage:
    from scripts.multihop_reasoning import MultiHopReasoningSystem
    
    reasoning_system = MultiHopReasoningSystem()
    result = reasoning_system.reason(query, max_hops=3)
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Set
import os
import json
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class ReasoningStep:
    """Represents a single step in multi-hop reasoning."""
    step_id: int
    query: str
    retrieved_chunks: List[str]
    reasoning: str
    confidence: float
    evidence: List[str]


@dataclass
class ReasoningPath:
    """Represents a complete reasoning path with multiple steps."""
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float
    path_score: float


class KnowledgeGraph:
    """Graph-based representation of medical knowledge chunks."""
    
    def __init__(self, chunks: List[str], embedder: SentenceTransformer):
        self.chunks = chunks
        self.embedder = embedder
        self.graph = nx.Graph()
        self.chunk_embeddings = None
        self.similarity_threshold = 0.7
        
        # Build the knowledge graph
        self._build_graph()
        
    def _build_graph(self):
        """Build the knowledge graph with semantic relationships."""
        print("Building knowledge graph...")
        
        # Add nodes for each chunk
        for i, chunk in enumerate(self.chunks):
            self.graph.add_node(i, chunk=chunk, chunk_id=i)
        
        # Compute chunk embeddings
        print("Computing chunk embeddings...")
        self.chunk_embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        
        # Add edges based on semantic similarity
        print("Adding semantic edges...")
        self._add_semantic_edges()
        
        # Add edges based on entity relationships
        print("Adding entity-based edges...")
        self._add_entity_edges()
        
        print(f"Knowledge graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _add_semantic_edges(self):
        """Add edges based on semantic similarity between chunks."""
        # Compute pairwise similarities
        similarities = np.dot(self.chunk_embeddings, self.chunk_embeddings.T)
        
        for i in range(len(self.chunks)):
            for j in range(i + 1, len(self.chunks)):
                similarity = similarities[i, j]
                if similarity > self.similarity_threshold:
                    self.graph.add_edge(i, j, similarity=similarity, edge_type='semantic')
    
    def _add_entity_edges(self):
        """Add edges based on shared medical entities."""
        # Extract medical entities from chunks
        entity_chunks = {}
        
        for i, chunk in enumerate(self.chunks):
            entities = self._extract_medical_entities(chunk)
            for entity in entities:
                if entity not in entity_chunks:
                    entity_chunks[entity] = []
                entity_chunks[entity].append(i)
        
        # Connect chunks that share entities
        for entity, chunk_indices in entity_chunks.items():
            if len(chunk_indices) > 1:
                for i in range(len(chunk_indices)):
                    for j in range(i + 1, len(chunk_indices)):
                        chunk_i, chunk_j = chunk_indices[i], chunk_indices[j]
                        if not self.graph.has_edge(chunk_i, chunk_j):
                            self.graph.add_edge(chunk_i, chunk_j, 
                                              shared_entity=entity, 
                                              edge_type='entity')
    
    def _extract_medical_entities(self, text: str) -> Set[str]:
        """Extract medical entities from text."""
        # Simple entity extraction - in practice, use NER models
        medical_terms = [
            'diabetes', 'hypertension', 'cancer', 'heart', 'lung', 'brain',
            'blood', 'pressure', 'sugar', 'insulin', 'treatment', 'therapy',
            'surgery', 'medication', 'drug', 'disease', 'symptom', 'diagnosis',
            'patient', 'clinical', 'medical', 'health', 'condition', 'disorder'
        ]
        
        entities = set()
        text_lower = text.lower()
        
        for term in medical_terms:
            if term in text_lower:
                entities.add(term)
        
        # Extract capitalized medical terms
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update([term.lower() for term in capitalized_terms])
        
        return entities
    
    def get_neighbors(self, chunk_id: int, max_neighbors: int = 10) -> List[int]:
        """Get neighboring chunks in the knowledge graph."""
        if chunk_id not in self.graph:
            return []
        
        neighbors = list(self.graph.neighbors(chunk_id))
        # Sort by edge weight (similarity or entity strength)
        neighbor_weights = []
        for neighbor in neighbors:
            edge_data = self.graph[chunk_id][neighbor]
            if 'similarity' in edge_data:
                weight = edge_data['similarity']
            else:
                weight = 1.0  # Entity-based edges
            neighbor_weights.append((neighbor, weight))
        
        neighbor_weights.sort(key=lambda x: x[1], reverse=True)
        return [neighbor for neighbor, _ in neighbor_weights[:max_neighbors]]
    
    def get_path(self, start_chunk: int, end_chunk: int, max_length: int = 3) -> List[int]:
        """Get the shortest path between two chunks."""
        try:
            path = nx.shortest_path(self.graph, start_chunk, end_chunk)
            if len(path) <= max_length:
                return path
            else:
                return path[:max_length]
        except nx.NetworkXNoPath:
            return []


class QueryExpander:
    """Expands queries based on retrieved information for multi-hop reasoning."""
    
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        
    def expand_query(self, original_query: str, retrieved_chunks: List[str], 
                    reasoning_context: str = "") -> List[str]:
        """Generate expanded queries for multi-hop reasoning."""
        expanded_queries = [original_query]
        
        # Extract key concepts from retrieved chunks
        key_concepts = self._extract_key_concepts(retrieved_chunks)
        
        # Generate follow-up queries
        for concept in key_concepts[:3]:  # Limit to top 3 concepts
            if concept.lower() not in original_query.lower():
                expanded_query = f"{original_query} related to {concept}"
                expanded_queries.append(expanded_query)
        
        # Generate reasoning-based queries
        if reasoning_context:
            reasoning_query = f"{original_query} considering {reasoning_context}"
            expanded_queries.append(reasoning_query)
        
        return expanded_queries
    
    def _extract_key_concepts(self, chunks: List[str]) -> List[str]:
        """Extract key medical concepts from chunks."""
        # Simple concept extraction - in practice, use more sophisticated methods
        concepts = []
        
        for chunk in chunks:
            # Extract medical terms
            medical_terms = re.findall(r'\b(?:diabetes|hypertension|cancer|heart|lung|brain|blood|pressure|sugar|insulin|treatment|therapy|surgery|medication|drug|disease|symptom|diagnosis|patient|clinical|medical|health|condition|disorder)\b', chunk.lower())
            concepts.extend(medical_terms)
        
        # Count frequency and return most common
        concept_counts = defaultdict(int)
        for concept in concepts:
            concept_counts[concept] += 1
        
        return [concept for concept, count in sorted(concept_counts.items(), 
                                                   key=lambda x: x[1], reverse=True)]


class ReasoningEngine:
    """Core reasoning engine for multi-hop inference."""
    
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.query_expander = QueryExpander(embedder)
        
    def reason_step(self, query: str, available_chunks: List[str], 
                   previous_steps: List[ReasoningStep] = None) -> ReasoningStep:
        """Perform a single reasoning step."""
        if previous_steps is None:
            previous_steps = []
        
        # Retrieve relevant chunks for this step
        relevant_chunks = self._retrieve_relevant_chunks(query, available_chunks)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query, relevant_chunks, previous_steps)
        
        # Calculate confidence
        confidence = self._calculate_confidence(query, relevant_chunks, reasoning)
        
        # Extract evidence
        evidence = self._extract_evidence(relevant_chunks, reasoning)
        
        return ReasoningStep(
            step_id=len(previous_steps) + 1,
            query=query,
            retrieved_chunks=relevant_chunks,
            reasoning=reasoning,
            confidence=confidence,
            evidence=evidence
        )
    
    def _retrieve_relevant_chunks(self, query: str, available_chunks: List[str]) -> List[str]:
        """Retrieve chunks relevant to the query."""
        if not available_chunks:
            return []
        
        # Encode query and chunks
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        chunk_embs = self.embedder.encode(available_chunks, convert_to_numpy=True)
        
        # Calculate similarities
        similarities = np.dot(query_emb, chunk_embs.T)[0]
        
        # Get top relevant chunks
        top_indices = np.argsort(similarities)[::-1][:5]
        return [available_chunks[i] for i in top_indices if similarities[i] > 0.3]
    
    def _generate_reasoning(self, query: str, chunks: List[str], 
                          previous_steps: List[ReasoningStep]) -> str:
        """Generate reasoning text based on query and chunks."""
        if not chunks:
            return "No relevant information found."
        
        # Simple reasoning generation - in practice, use LLM
        reasoning_parts = []
        
        for i, chunk in enumerate(chunks[:3]):  # Use top 3 chunks
            reasoning_parts.append(f"From source {i+1}: {chunk[:100]}...")
        
        if previous_steps:
            reasoning_parts.append(f"Building on previous findings: {len(previous_steps)} steps completed.")
        
        return " ".join(reasoning_parts)
    
    def _calculate_confidence(self, query: str, chunks: List[str], reasoning: str) -> float:
        """Calculate confidence score for the reasoning step."""
        if not chunks:
            return 0.0
        
        # Simple confidence calculation based on chunk relevance
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        chunk_embs = self.embedder.encode(chunks, convert_to_numpy=True)
        
        similarities = np.dot(query_emb, chunk_embs.T)[0]
        avg_similarity = np.mean(similarities)
        
        # Boost confidence if we have multiple relevant chunks
        chunk_bonus = min(len(chunks) * 0.1, 0.3)
        
        return min(avg_similarity + chunk_bonus, 1.0)
    
    def _extract_evidence(self, chunks: List[str], reasoning: str) -> List[str]:
        """Extract evidence statements from chunks."""
        evidence = []
        for chunk in chunks:
            # Extract key sentences as evidence
            sentences = chunk.split('.')
            for sentence in sentences[:2]:  # Take first 2 sentences
                if len(sentence.strip()) > 20:  # Filter short sentences
                    evidence.append(sentence.strip())
        return evidence


class MultiHopReasoningSystem:
    """Main multi-hop reasoning system integrating with RETRO model."""
    
    def __init__(self, corpus_path: str = None, index_path: str = None, 
                 embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Get project paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        self.corpus_path = corpus_path or os.path.join(project_root, "data", "processed_chunks", "full_corpus.pkl")
        self.index_path = index_path or os.path.join(project_root, "faiss_index", "medical_faiss.index")
        
        # Load components
        self._load_components(embedder_model)
        
        # Initialize reasoning components
        self.reasoning_engine = ReasoningEngine(self.embedder)
        self.knowledge_graph = KnowledgeGraph(self.full_corpus, self.embedder)
        
    def _load_components(self, embedder_model: str):
        """Load the retrieval system components."""
        print("Loading multi-hop reasoning components...")
        
        # Load corpus
        with open(self.corpus_path, "rb") as f:
            self.full_corpus = pickle.load(f)
            
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load embedder
        self.embedder = SentenceTransformer(embedder_model)
        
        print(f"Components loaded. Corpus size: {len(self.full_corpus)}")
    
    def reason(self, query: str, max_hops: int = 3, top_k: int = 10) -> ReasoningPath:
        """Perform multi-hop reasoning for a given query."""
        print(f"Starting multi-hop reasoning for: '{query}'")
        print(f"Max hops: {max_hops}, Top-k: {top_k}")
        
        reasoning_steps = []
        current_chunks = set()
        
        # Initial retrieval
        initial_chunks = self._initial_retrieval(query, top_k)
        current_chunks.update(initial_chunks)
        
        # First reasoning step
        step = self.reasoning_engine.reason_step(query, list(current_chunks))
        reasoning_steps.append(step)
        
        # Multi-hop reasoning
        for hop in range(1, max_hops):
            print(f"Hop {hop + 1}: Expanding reasoning...")
            
            # Expand query based on previous reasoning
            expanded_queries = self.reasoning_engine.query_expander.expand_query(
                query, step.retrieved_chunks, step.reasoning
            )
            
            # Get additional chunks through graph traversal
            new_chunks = self._get_related_chunks(step, current_chunks)
            current_chunks.update(new_chunks)
            
            # Perform next reasoning step
            if new_chunks:
                next_query = expanded_queries[1] if len(expanded_queries) > 1 else query
                step = self.reasoning_engine.reason_step(
                    next_query, list(current_chunks), reasoning_steps
                )
                reasoning_steps.append(step)
            else:
                print(f"No new chunks found at hop {hop + 1}, stopping.")
                break
        
        # Generate final answer
        final_answer = self._generate_final_answer(reasoning_steps)
        
        # Calculate path confidence
        path_confidence = self._calculate_path_confidence(reasoning_steps)
        path_score = self._calculate_path_score(reasoning_steps)
        
        return ReasoningPath(
            steps=reasoning_steps,
            final_answer=final_answer,
            confidence=path_confidence,
            path_score=path_score
        )
    
    def _initial_retrieval(self, query: str, top_k: int) -> List[int]:
        """Perform initial retrieval using FAISS index."""
        query_emb = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_emb, top_k)
        return indices[0].tolist()
    
    def _get_related_chunks(self, step: ReasoningStep, current_chunks: Set[int]) -> List[int]:
        """Get related chunks through graph traversal."""
        new_chunks = []
        
        for chunk_id in current_chunks:
            # Get neighbors from knowledge graph
            neighbors = self.knowledge_graph.get_neighbors(chunk_id, max_neighbors=5)
            
            for neighbor in neighbors:
                if neighbor not in current_chunks:
                    new_chunks.append(neighbor)
        
        return new_chunks[:10]  # Limit new chunks
    
    def _generate_final_answer(self, steps: List[ReasoningStep]) -> str:
        """Generate final answer from reasoning steps."""
        if not steps:
            return "No reasoning steps completed."
        
        # Combine evidence from all steps
        all_evidence = []
        for step in steps:
            all_evidence.extend(step.evidence)
        
        # Simple answer generation - in practice, use LLM
        if all_evidence:
            # Take the most relevant evidence
            key_evidence = all_evidence[:3]
            answer = f"Based on multi-hop reasoning across {len(steps)} steps: " + \
                    " ".join(key_evidence)
        else:
            answer = f"Multi-hop reasoning completed in {len(steps)} steps, " + \
                    "but no conclusive evidence found."
        
        return answer
    
    def _calculate_path_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence for the reasoning path."""
        if not steps:
            return 0.0
        
        confidences = [step.confidence for step in steps]
        return np.mean(confidences)
    
    def _calculate_path_score(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall score for the reasoning path."""
        if not steps:
            return 0.0
        
        # Combine confidence and evidence quality
        confidence_score = self._calculate_path_confidence(steps)
        evidence_score = min(len([step for step in steps if step.evidence]) / len(steps), 1.0)
        
        return (confidence_score + evidence_score) / 2
    
    def compare_with_single_hop(self, query: str) -> Dict:
        """Compare multi-hop reasoning with single-hop retrieval."""
        print("Comparing multi-hop vs single-hop reasoning...")
        
        # Single-hop retrieval
        single_hop_chunks = self._initial_retrieval(query, 5)
        single_hop_text = [self.full_corpus[i] for i in single_hop_chunks]
        
        # Multi-hop reasoning
        multi_hop_path = self.reason(query, max_hops=3)
        
        return {
            'query': query,
            'single_hop': {
                'chunks': single_hop_text,
                'num_chunks': len(single_hop_text)
            },
            'multi_hop': {
                'path': multi_hop_path,
                'num_steps': len(multi_hop_path.steps),
                'confidence': multi_hop_path.confidence,
                'path_score': multi_hop_path.path_score
            }
        }


def main():
    """Demonstration of multi-hop reasoning system."""
    print("Multi-Hop Reasoning System Demo")
    print("=" * 50)
    
    # Initialize system
    reasoning_system = MultiHopReasoningSystem()
    
    # Example queries for multi-hop reasoning
    test_queries = [
        "What are the treatment options for diabetes and how do they relate to blood pressure management?",
        "How does heart disease connect to diabetes and what are the combined treatment approaches?",
        "What are the symptoms of hypertension and how do they differ from diabetes symptoms?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        # Perform multi-hop reasoning
        result = reasoning_system.compare_with_single_hop(query)
        
        print(f"\nSINGLE-HOP RETRIEVAL:")
        print(f"Number of chunks: {result['single_hop']['num_chunks']}")
        for i, chunk in enumerate(result['single_hop']['chunks'][:2]):
            print(f"  {i+1}. {chunk[:100]}...")
        
        print(f"\nMULTI-HOP REASONING:")
        print(f"Number of steps: {result['multi_hop']['num_steps']}")
        print(f"Confidence: {result['multi_hop']['confidence']:.3f}")
        print(f"Path score: {result['multi_hop']['path_score']:.3f}")
        
        path = result['multi_hop']['path']
        for step in path.steps:
            print(f"\n  Step {step.step_id}: {step.query}")
            print(f"    Reasoning: {step.reasoning[:100]}...")
            print(f"    Confidence: {step.confidence:.3f}")
        
        print(f"\n  Final Answer: {path.final_answer}")


if __name__ == "__main__":
    main()
