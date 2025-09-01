#!/usr/bin/env python3
"""
Demo script for Hybrid Retrieval System
=======================================

This script demonstrates the key features of the hybrid search system
with example queries and results.
"""

import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from hybrid_search import HybridRetrievalSystem

def print_separator(title):
    """Print a formatted separator."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_results(results, method_name, max_results=3):
    """Print search results in a formatted way."""
    print(f"\n🔍 {method_name} Results:")
    print("-" * 40)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results[:max_results], 1):
        print(f"\n{i}. {result.chunk_id}")
        print(f"   📄 Source: {result.source_name}")
        print(f"   📊 Score: {getattr(result, f'{method_name.lower().replace(" ", "_")}_score', result.hybrid_score):.3f}")
        print(f"   📝 Text: {result.text[:150]}...")
        if len(result.text) > 150:
            print("      ...")

def demo_hybrid_search():
    """Demonstrate the hybrid search system."""
    
    print("🚀 Hybrid Retrieval System Demo")
    print("This demo shows how the hybrid system combines vector and keyword search")
    
    # Initialize system
    print_separator("INITIALIZING SYSTEM")
    system = HybridRetrievalSystem(db_path="demo_hybrid_index.db")
    
    # Check if we need to build the index
    if not Path("demo_hybrid_index.db").exists():
        print("📚 Building hybrid index from Week 4 data...")
        success = system.build_hybrid_index()
        if not success:
            print("❌ Failed to build hybrid index")
            print("Please ensure Week 4 data exists in ../HomeWork4/")
            return
        print("✅ Hybrid index built successfully!")
    else:
        print("📚 Loading existing hybrid index...")
        if not system.load_existing_index():
            print("❌ Failed to load existing index")
            return
        print("✅ Hybrid index loaded successfully!")
    
    print(f"📊 System loaded with {system.faiss_index.ntotal} vectors")
    print(f"🤖 Model: {system.metadata['model']}")
    
    # Demo queries
    demo_queries = [
        {
            "query": "machine learning algorithms",
            "description": "Broad semantic concept - should benefit from vector search"
        },
        {
            "query": "BERT",
            "description": "Specific model name - should benefit from keyword search"
        },
        {
            "query": "attention mechanism in transformers",
            "description": "Mixed query - should benefit from hybrid approach"
        }
    ]
    
    for i, query_data in enumerate(demo_queries, 1):
        query = query_data["query"]
        description = query_data["description"]
        
        print_separator(f"QUERY {i}: {query}")
        print(f"📝 Description: {description}")
        
        # Run all search methods
        print("\n⏱️  Running searches...")
        
        # Vector search
        start_time = time.time()
        vector_results = system.vector_search(query, k=5)
        vector_time = (time.time() - start_time) * 1000
        
        # Keyword search
        start_time = time.time()
        keyword_results = system.keyword_search(query, k=5)
        keyword_time = (time.time() - start_time) * 1000
        
        # Hybrid search (weighted)
        start_time = time.time()
        hybrid_weighted = system.hybrid_search(query, k=5, alpha=0.6, fusion_method="weighted")
        hybrid_weighted_time = (time.time() - start_time) * 1000
        
        # Hybrid search (RRF)
        start_time = time.time()
        hybrid_rrf = system.hybrid_search(query, k=5, fusion_method="rrf")
        hybrid_rrf_time = (time.time() - start_time) * 1000
        
        # Display timing information
        print(f"\n⏱️  Search Times:")
        print(f"   Vector:     {vector_time:.1f} ms")
        print(f"   Keyword:    {keyword_time:.1f} ms")
        print(f"   Hybrid-W:   {hybrid_weighted_time:.1f} ms")
        print(f"   Hybrid-RRF: {hybrid_rrf_time:.1f} ms")
        
        # Display results
        print_results(vector_results, "Vector", max_results=3)
        print_results(keyword_results, "Keyword", max_results=3)
        print_results(hybrid_weighted, "Hybrid Weighted", max_results=3)
        print_results(hybrid_rrf, "Hybrid RRF", max_results=3)
        
        # Analyze result overlap
        vector_chunks = set([r.chunk_id for r in vector_results])
        keyword_chunks = set([r.chunk_id for r in keyword_results])
        hybrid_weighted_chunks = set([r.chunk_id for r in hybrid_weighted])
        hybrid_rrf_chunks = set([r.chunk_id for r in hybrid_rrf])
        
        print(f"\n📊 Result Analysis:")
        print(f"   Vector results:     {len(vector_chunks)} unique chunks")
        print(f"   Keyword results:    {len(keyword_chunks)} unique chunks")
        print(f"   Hybrid-W results:   {len(hybrid_weighted_chunks)} unique chunks")
        print(f"   Hybrid-RRF results: {len(hybrid_rrf_chunks)} unique chunks")
        
        # Calculate overlaps
        vector_keyword_overlap = len(vector_chunks.intersection(keyword_chunks))
        print(f"   Vector-Keyword overlap: {vector_keyword_overlap} chunks")
        
        # Show which method found unique results
        vector_unique = vector_chunks - keyword_chunks
        keyword_unique = keyword_chunks - vector_chunks
        
        if vector_unique:
            print(f"   Vector-only results: {len(vector_unique)} chunks")
        if keyword_unique:
            print(f"   Keyword-only results: {len(keyword_unique)} chunks")
    
    # Summary
    print_separator("DEMO SUMMARY")
    print("🎯 Key Takeaways:")
    print("   • Vector search excels at semantic understanding")
    print("   • Keyword search provides exact matches")
    print("   • Hybrid methods combine both strengths")
    print("   • Different fusion strategies have different characteristics")
    print("   • RRF fusion can find results that individual methods miss")
    
    print("\n💡 When to use each method:")
    print("   • Vector search: Conceptual queries, synonyms, related topics")
    print("   • Keyword search: Exact terms, proper nouns, specific phrases")
    print("   • Hybrid search: General-purpose, when you want both approaches")
    
    print("\n🔧 Configuration tips:")
    print("   • Adjust alpha parameter (0.6) to weight vector vs keyword search")
    print("   • Use RRF for rank-based fusion, weighted for score-based fusion")
    print("   • Increase k parameter to get more diverse results")
    
    # Cleanup
    print(f"\n🧹 Cleaning up demo database...")
    try:
        Path("demo_hybrid_index.db").unlink()
        print("✅ Demo database removed")
    except:
        print("⚠️  Could not remove demo database")

if __name__ == "__main__":
    demo_hybrid_search()
