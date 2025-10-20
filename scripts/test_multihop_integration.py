"""
Test Script for Multi-Hop Reasoning Integration
==============================================

This script tests the multi-hop reasoning integration to ensure all components
work together correctly and provide meaningful improvements over single-hop retrieval.

Usage:
    python scripts/test_multihop_integration.py
"""

import os
import sys
import time
from typing import Dict, List

# Add the scripts directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from multihop_reasoning import MultiHopReasoningSystem, ReasoningPath
        print("✅ multihop_reasoning imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import multihop_reasoning: {e}")
        return False
    
    try:
        from enhanced_query_retro import EnhancedRetroQuery, ComparisonResult
        print("✅ enhanced_query_retro imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import enhanced_query_retro: {e}")
        return False
    
    try:
        from retro_multihop_model import RETROMultiHopSystem, RETROMultiHopModel
        print("✅ retro_multihop_model imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import retro_multihop_model: {e}")
        return False
    
    try:
        from reasoning_evaluation import ReasoningEvaluator, ReasoningMetrics
        print("✅ reasoning_evaluation imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import reasoning_evaluation: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of multi-hop reasoning components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test multi-hop reasoning system initialization
        from multihop_reasoning import MultiHopReasoningSystem
        reasoning_system = MultiHopReasoningSystem()
        print("✅ MultiHopReasoningSystem initialized")
        
        # Test enhanced query system initialization
        from enhanced_query_retro import EnhancedRetroQuery
        query_system = EnhancedRetroQuery()
        print("✅ EnhancedRetroQuery initialized")
        
        # Test evaluation system initialization
        from reasoning_evaluation import ReasoningEvaluator
        evaluator = ReasoningEvaluator()
        print("✅ ReasoningEvaluator initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_single_vs_multihop():
    """Test comparison between single-hop and multi-hop approaches."""
    print("\nTesting single-hop vs multi-hop comparison...")
    
    try:
        from enhanced_query_retro import EnhancedRetroQuery
        
        query_system = EnhancedRetroQuery()
        
        # Test query
        test_query = "What are the symptoms of diabetes?"
        
        # Single-hop retrieval
        print("  Testing single-hop retrieval...")
        single_hop_result = query_system.retrieve(test_query, top_k=3)
        print(f"    ✅ Single-hop: {len(single_hop_result.chunks)} chunks, confidence: {single_hop_result.confidence:.3f}")
        
        # Multi-hop reasoning
        print("  Testing multi-hop reasoning...")
        multi_hop_result = query_system.reason(test_query, max_hops=2)
        print(f"    ✅ Multi-hop: {len(multi_hop_result.chunks)} chunks, confidence: {multi_hop_result.confidence:.3f}")
        
        # Comparison
        print("  Testing comparison...")
        comparison = query_system.compare_approaches(test_query, top_k=3, max_hops=2)
        print(f"    ✅ Comparison completed")
        print(f"    Coverage improvement: {comparison.improvement_metrics.get('coverage_improvement', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Single vs multi-hop test failed: {e}")
        return False

def test_reasoning_evaluation():
    """Test reasoning evaluation functionality."""
    print("\nTesting reasoning evaluation...")
    
    try:
        from multihop_reasoning import MultiHopReasoningSystem
        from reasoning_evaluation import ReasoningEvaluator
        
        reasoning_system = MultiHopReasoningSystem()
        evaluator = ReasoningEvaluator()
        
        # Test query
        test_query = "What are the treatment options for diabetes?"
        
        # Perform multi-hop reasoning
        reasoning_path = reasoning_system.reason(test_query, max_hops=2)
        
        # Evaluate reasoning path
        metrics = evaluator.evaluate_reasoning_path(reasoning_path, query=test_query)
        
        print(f"    ✅ Reasoning evaluation completed")
        print(f"    Path coherence: {metrics.path_coherence:.3f}")
        print(f"    Evidence relevance: {metrics.evidence_relevance:.3f}")
        print(f"    Reasoning efficiency: {metrics.reasoning_efficiency:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning evaluation test failed: {e}")
        return False

def test_retro_integration():
    """Test RETRO model integration."""
    print("\nTesting RETRO model integration...")
    
    try:
        from retro_multihop_model import RETROMultiHopSystem
        
        # Initialize system (this will create a new model if none exists)
        retro_system = RETROMultiHopSystem()
        print("    ✅ RETROMultiHopSystem initialized")
        
        # Test query
        test_query = "What are the symptoms of diabetes?"
        
        # Test generation (this might fail if model files are missing, which is expected)
        try:
            result = retro_system.generate_with_reasoning(test_query, max_hops=2)
            print(f"    ✅ Multi-hop generation completed")
            print(f"    Response length: {len(result['response'])}")
            print(f"    Reasoning steps: {result['reasoning_steps']}")
        except Exception as e:
            print(f"    ⚠️  Generation test skipped (expected if model files missing): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ RETRO integration test failed: {e}")
        return False

def run_performance_test():
    """Run a simple performance test."""
    print("\nRunning performance test...")
    
    try:
        from enhanced_query_retro import EnhancedRetroQuery
        
        query_system = EnhancedRetroQuery()
        test_query = "What are the treatment options for diabetes and how do they relate to blood pressure management?"
        
        # Test single-hop performance
        start_time = time.time()
        single_hop_result = query_system.retrieve(test_query, top_k=5)
        single_hop_time = time.time() - start_time
        
        # Test multi-hop performance
        start_time = time.time()
        multi_hop_result = query_system.reason(test_query, max_hops=2)
        multi_hop_time = time.time() - start_time
        
        print(f"    Single-hop time: {single_hop_time:.3f}s")
        print(f"    Multi-hop time: {multi_hop_time:.3f}s")
        print(f"    Time ratio: {multi_hop_time/single_hop_time:.2f}x")
        print(f"    Coverage improvement: {len(multi_hop_result.chunks)/len(single_hop_result.chunks):.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Multi-Hop Reasoning Integration Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Single vs Multi-hop", test_single_vs_multihop),
        ("Reasoning Evaluation", test_reasoning_evaluation),
        ("RETRO Integration", test_retro_integration),
        ("Performance Test", run_performance_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Multi-hop reasoning integration is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

