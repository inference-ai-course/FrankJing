#!/usr/bin/env python3
"""
Test script for Hybrid Retrieval System
=======================================

This script tests the basic functionality of the hybrid search system
to ensure everything is working correctly.
"""

import sys
import os
import time
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from hybrid_search import HybridRetrievalSystem

def test_system():
    """Test the hybrid retrieval system."""
    
    print("🧪 Testing Hybrid Retrieval System")
    print("=" * 40)
    
    # Test 1: Initialize system
    print("\n1️⃣ Testing system initialization...")
    try:
        system = HybridRetrievalSystem(db_path="test_hybrid_index.db")
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    # Test 2: Check if Week 4 data exists
    print("\n2️⃣ Checking Week 4 data...")
    week4_path = Path("../HomeWork4")
    required_files = ["faiss.index", "embeddings.npy", "chunks.jsonl", "meta.json"]
    
    missing_files = []
    for file in required_files:
        if not (week4_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing Week 4 files: {missing_files}")
        print("Please ensure Week 4 data exists in ../HomeWork4/")
        return False
    else:
        print("✅ All Week 4 files found")
    
    # Test 3: Build hybrid index
    print("\n3️⃣ Testing hybrid index building...")
    try:
        success = system.build_hybrid_index()
        if success:
            print("✅ Hybrid index built successfully")
        else:
            print("❌ Failed to build hybrid index")
            return False
    except Exception as e:
        print(f"❌ Index building failed: {e}")
        return False
    
    # Test 4: Test vector search
    print("\n4️⃣ Testing vector search...")
    try:
        results = system.vector_search("machine learning", k=2)
        if results:
            print(f"✅ Vector search returned {len(results)} results")
            print(f"   Top result: {results[0].chunk_id} (score: {results[0].vector_score:.3f})")
        else:
            print("❌ Vector search returned no results")
            return False
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return False
    
    # Test 5: Test keyword search
    print("\n5️⃣ Testing keyword search...")
    try:
        results = system.keyword_search("machine learning", k=2)
        if results:
            print(f"✅ Keyword search returned {len(results)} results")
            print(f"   Top result: {results[0].chunk_id} (score: {results[0].keyword_score:.3f})")
        else:
            print("❌ Keyword search returned no results")
            return False
    except Exception as e:
        print(f"❌ Keyword search failed: {e}")
        return False
    
    # Test 6: Test hybrid search (weighted)
    print("\n6️⃣ Testing hybrid search (weighted)...")
    try:
        results = system.hybrid_search("machine learning", k=2, alpha=0.6, fusion_method="weighted")
        if results:
            print(f"✅ Hybrid search (weighted) returned {len(results)} results")
            print(f"   Top result: {results[0].chunk_id} (hybrid: {results[0].hybrid_score:.3f})")
        else:
            print("❌ Hybrid search (weighted) returned no results")
            return False
    except Exception as e:
        print(f"❌ Hybrid search (weighted) failed: {e}")
        return False
    
    # Test 7: Test hybrid search (RRF)
    print("\n7️⃣ Testing hybrid search (RRF)...")
    try:
        results = system.hybrid_search("machine learning", k=2, fusion_method="rrf")
        if results:
            print(f"✅ Hybrid search (RRF) returned {len(results)} results")
            print(f"   Top result: {results[0].chunk_id} (hybrid: {results[0].hybrid_score:.3f})")
        else:
            print("❌ Hybrid search (RRF) returned no results")
            return False
    except Exception as e:
        print(f"❌ Hybrid search (RRF) failed: {e}")
        return False
    
    # Test 8: Performance test
    print("\n8️⃣ Testing performance...")
    try:
        queries = ["machine learning", "neural networks", "transformer"]
        times = []
        
        for query in queries:
            start_time = time.time()
            system.hybrid_search(query, k=3)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"✅ Performance test completed")
        print(f"   Average search time: {avg_time:.1f} ms")
        
        if avg_time > 1000:  # More than 1 second
            print("⚠️  Warning: Search time is quite slow")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    
    # Test 9: Database integrity
    print("\n9️⃣ Testing database integrity...")
    try:
        import sqlite3
        conn = sqlite3.connect("test_hybrid_index.db")
        
        # Check document count
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        
        # Check chunk count
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        
        # Check FTS5 table
        fts_count = conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
        
        conn.close()
        
        print(f"✅ Database integrity check passed")
        print(f"   Documents: {doc_count}")
        print(f"   Chunks: {chunk_count}")
        print(f"   FTS5 entries: {fts_count}")
        
        if doc_count == 0 or chunk_count == 0:
            print("❌ Database appears to be empty")
            return False
            
    except Exception as e:
        print(f"❌ Database integrity check failed: {e}")
        return False
    
    # Cleanup
    print("\n🧹 Cleaning up test database...")
    try:
        os.remove("test_hybrid_index.db")
        print("✅ Test database removed")
    except:
        print("⚠️  Could not remove test database")
    
    print("\n🎉 All tests passed! The hybrid retrieval system is working correctly.")
    return True

if __name__ == "__main__":
    success = test_system()
    if success:
        print("\n✅ System is ready for use!")
        print("\nNext steps:")
        print("1. Build the full index: python hybrid_search.py build")
        print("2. Run evaluation: python simple_evaluation.py")
        print("3. Start API server: python fastapi_hybrid.py")
    else:
        print("\n❌ System tests failed. Please check the errors above.")
        sys.exit(1)
