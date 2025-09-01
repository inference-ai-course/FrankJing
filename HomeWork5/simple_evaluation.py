#!/usr/bin/env python3
"""
Simple evaluation script for Hybrid Retrieval System
====================================================

This script provides a quick evaluation of the hybrid search system
without requiring Jupyter notebook dependencies.
"""

import sys
import json
import time
from typing import List, Dict, Any
import numpy as np

# Add current directory to path
sys.path.append('.')

from hybrid_search import HybridRetrievalSystem

def run_simple_evaluation():
    """Run a simple evaluation of the hybrid search system."""
    
    print("🚀 Starting Hybrid Retrieval System Evaluation")
    print("=" * 50)
    
    # Initialize system
    system = HybridRetrievalSystem(db_path="hybrid_index.db")
    
    # Build hybrid index
    print("📚 Building hybrid index...")
    success = system.build_hybrid_index()
    
    if not success:
        print("❌ Failed to build hybrid index")
        print("Please ensure Week 4 data exists in ../HomeWork4/")
        return
    
    print("✅ Hybrid index built successfully!")
    print(f"📊 System loaded with {system.faiss_index.ntotal} vectors")
    print(f"🤖 Model: {system.metadata['model']}")
    
    # Define test queries
    test_queries = [
        "machine learning algorithms",
        "neural networks", 
        "transformer architecture",
        "BERT",
        "attention mechanism",
        "natural language processing",
        "word embeddings",
        "GPT",
        "contrastive learning",
        "fine-tuning",
        "retrieval augmented generation",
        "vector database"
    ]
    
    print(f"\n🔍 Testing {len(test_queries)} queries...")
    
    # Run evaluation
    results = {
        'vector': {'times': [], 'scores': []},
        'keyword': {'times': [], 'scores': []},
        'hybrid_weighted': {'times': [], 'scores': []},
        'hybrid_rrf': {'times': [], 'scores': []}
    }
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}/{len(test_queries)}: {query}")
        
        # Vector search
        start_time = time.time()
        vector_results = system.vector_search(query, k=3)
        vector_time = (time.time() - start_time) * 1000
        vector_score = np.mean([r.vector_score for r in vector_results]) if vector_results else 0
        
        # Keyword search
        start_time = time.time()
        keyword_results = system.keyword_search(query, k=3)
        keyword_time = (time.time() - start_time) * 1000
        keyword_score = np.mean([r.keyword_score for r in keyword_results]) if keyword_results else 0
        
        # Hybrid search (weighted)
        start_time = time.time()
        hybrid_weighted = system.hybrid_search(query, k=3, alpha=0.6, fusion_method="weighted")
        hybrid_weighted_time = (time.time() - start_time) * 1000
        hybrid_weighted_score = np.mean([r.hybrid_score for r in hybrid_weighted]) if hybrid_weighted else 0
        
        # Hybrid search (RRF)
        start_time = time.time()
        hybrid_rrf = system.hybrid_search(query, k=3, fusion_method="rrf")
        hybrid_rrf_time = (time.time() - start_time) * 1000
        hybrid_rrf_score = np.mean([r.hybrid_score for r in hybrid_rrf]) if hybrid_rrf else 0
        
        # Store results
        results['vector']['times'].append(vector_time)
        results['vector']['scores'].append(vector_score)
        results['keyword']['times'].append(keyword_time)
        results['keyword']['scores'].append(keyword_score)
        results['hybrid_weighted']['times'].append(hybrid_weighted_time)
        results['hybrid_weighted']['scores'].append(hybrid_weighted_score)
        results['hybrid_rrf']['times'].append(hybrid_rrf_time)
        results['hybrid_rrf']['scores'].append(hybrid_rrf_score)
        
        print(f"  ⏱️  Times: Vector={vector_time:.1f}ms, Keyword={keyword_time:.1f}ms, "
              f"Hybrid-W={hybrid_weighted_time:.1f}ms, Hybrid-RRF={hybrid_rrf_time:.1f}ms")
        print(f"  📊 Scores: Vector={vector_score:.3f}, Keyword={keyword_score:.3f}, "
              f"Hybrid-W={hybrid_weighted_score:.3f}, Hybrid-RRF={hybrid_rrf_score:.3f}")
    
    # Calculate summary statistics
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 50)
    
    for method, data in results.items():
        avg_time = np.mean(data['times'])
        avg_score = np.mean(data['scores'])
        std_time = np.std(data['times'])
        std_score = np.std(data['scores'])
        
        print(f"\n{method.upper()}:")
        print(f"  ⏱️  Average Time: {avg_time:.1f} ± {std_time:.1f} ms")
        print(f"  📊 Average Score: {avg_score:.3f} ± {std_score:.3f}")
    
    # Compare methods
    print(f"\n🏆 PERFORMANCE COMPARISON:")
    
    # Time comparison
    times = {method: np.mean(data['times']) for method, data in results.items()}
    fastest_method = min(times, key=times.get)
    print(f"  ⚡ Fastest Method: {fastest_method} ({times[fastest_method]:.1f} ms)")
    
    # Score comparison
    scores = {method: np.mean(data['scores']) for method, data in results.items()}
    best_method = max(scores, key=scores.get)
    print(f"  🎯 Best Scoring: {best_method} ({scores[best_method]:.3f})")
    
    # Hybrid vs individual methods
    hybrid_weighted_time = times['hybrid_weighted']
    hybrid_rrf_time = times['hybrid_rrf']
    vector_time = times['vector']
    keyword_time = times['keyword']
    
    print(f"\n🔄 HYBRID ANALYSIS:")
    print(f"  Weighted fusion overhead: {hybrid_weighted_time - (vector_time + keyword_time):.1f} ms")
    print(f"  RRF fusion overhead: {hybrid_rrf_time - (vector_time + keyword_time):.1f} ms")
    
    # Save results
    evaluation_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_queries': len(test_queries),
        'model': system.metadata['model'],
        'embedding_dim': system.metadata['dim'],
        'test_queries': test_queries,
        'results': results,
        'summary': {
            'times': times,
            'scores': scores,
            'fastest_method': fastest_method,
            'best_scoring_method': best_method
        }
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"\n💾 Results saved to 'evaluation_results.json'")
    print(f"✅ Evaluation completed successfully!")

if __name__ == "__main__":
    run_simple_evaluation()
