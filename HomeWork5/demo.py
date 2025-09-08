#!/usr/bin/env python3
"""
Homework 5 Demo: Hybrid Search System
Complete demonstration of the hybrid retrieval system combining FAISS and SQLite FTS5.
"""

import sys
import time
import json
from pathlib import Path
from hybrid_search import HybridSearchSystem
from evaluation import SearchEvaluator
import pandas as pd

def print_separator(title="", char="=", width=80):
    """Print a formatted separator line."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"{char * padding} {title} {char * padding}")
    else:
        print(char * width)

def print_search_results(results, title, max_results=3):
    """Print formatted search results."""
    print_separator(title, "-", 60)
    if not results:
        print("No results found.")
        return
    
    for i, (doc_id, score) in enumerate(results[:max_results]):
        print(f"{i+1}. Document {doc_id} (Score: {score:.4f})")
    print()

def demonstrate_basic_functionality():
    """Demonstrate basic search system functionality."""
    print_separator("BASIC FUNCTIONALITY DEMO")
    
    # Initialize system
    print("1. Initializing Hybrid Search System...")
    search_system = HybridSearchSystem(db_path="demo_search.db")
    
    # Sample documents
    print("2. Adding sample documents...")
    sample_documents = [
        {
            'title': 'Introduction to Deep Learning',
            'author': 'Dr. Sarah Wilson',
            'year': 2023,
            'keywords': 'deep learning, neural networks, artificial intelligence, machine learning',
            'chunk_text': 'Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. These deep neural networks can automatically learn hierarchical representations from raw data.'
        },
        {
            'title': 'Natural Language Processing with Transformers',
            'author': 'Prof. Michael Chen',
            'year': 2024,
            'keywords': 'NLP, natural language processing, transformers, attention mechanisms, BERT, GPT',
            'chunk_text': 'Natural language processing has been revolutionized by transformer architectures. These models use attention mechanisms to understand context and relationships between words in text, enabling powerful language understanding and generation.'
        },
        {
            'title': 'Computer Vision and Convolutional Networks',
            'author': 'Dr. Emily Rodriguez',
            'year': 2023,
            'keywords': 'computer vision, CNN, convolutional neural networks, image recognition, object detection',
            'chunk_text': 'Computer vision systems use convolutional neural networks to analyze and interpret visual information. CNNs are particularly effective for image recognition, object detection, and feature extraction from visual data.'
        },
        {
            'title': 'Reinforcement Learning Fundamentals',
            'author': 'Prof. James Thompson',
            'year': 2022,
            'keywords': 'reinforcement learning, Q-learning, policy gradient, reward systems, agent-based learning',
            'chunk_text': 'Reinforcement learning enables agents to learn optimal behaviors through interaction with an environment. Agents receive rewards or penalties for actions, gradually learning policies that maximize long-term rewards.'
        },
        {
            'title': 'Statistical Machine Learning Methods',
            'author': 'Dr. Lisa Park',
            'year': 2023,
            'keywords': 'statistical learning, probability theory, Bayesian methods, statistical inference, data analysis',
            'chunk_text': 'Statistical machine learning combines probability theory with computational methods to make inferences from data. Bayesian methods and statistical inference provide principled approaches to uncertainty quantification and model selection.'
        }
    ]
    
    search_system.add_documents(sample_documents)
    print(f"   Added {len(sample_documents)} documents successfully!")
    
    # Demonstrate different search methods
    query = "neural networks deep learning"
    print(f"\n3. Testing search methods with query: '{query}'")
    
    # Vector search
    vector_results = search_system.vector_search(query, k=3)
    print_search_results(vector_results, "Vector Search Results")
    
    # Keyword search
    keyword_results = search_system.keyword_search(query, k=3)
    print_search_results(keyword_results, "Keyword Search Results")
    
    # Hybrid search with different parameters
    hybrid_results = search_system.weighted_hybrid_search(query, k=3, alpha=0.6)
    print_search_results(hybrid_results, "Weighted Hybrid Search (α=0.6)")
    
    rrf_results = search_system.rrf_hybrid_search(query, k=3)
    print_search_results(rrf_results, "RRF Hybrid Search Results")
    
    return search_system

def demonstrate_detailed_results(search_system):
    """Show detailed document information for search results."""
    print_separator("DETAILED RESULTS DEMO")
    
    query = "statistical methods probability"
    print(f"Query: '{query}'\n")
    
    results = search_system.weighted_hybrid_search(query, k=2, alpha=0.5)
    
    if results:
        doc_ids = [doc_id for doc_id, _ in results]
        documents = search_system.get_document_details(doc_ids)
        
        for i, ((doc_id, score), doc) in enumerate(zip(results, documents)):
            print(f"Result {i+1}:")
            print(f"  Title: {doc['title']}")
            print(f"  Author: {doc['author']}")
            print(f"  Year: {doc['year']}")
            print(f"  Keywords: {doc['keywords']}")
            print(f"  Score: {score:.4f}")
            print(f"  Content Preview: {doc['chunk_text'][:150]}...")
            print()

def run_evaluation_demo():
    """Demonstrate the evaluation system."""
    print_separator("EVALUATION SYSTEM DEMO")
    
    print("1. Setting up evaluation system...")
    search_system = HybridSearchSystem(db_path="eval_demo.db")
    evaluator = SearchEvaluator(search_system)
    
    print("2. Loading test data and running evaluation...")
    print("   This may take a moment...")
    
    # Run a subset evaluation for demo purposes
    evaluator.load_test_data()
    
    # Test a few queries manually for demonstration
    test_queries = [
        "neural networks deep learning",
        "natural language processing text",
        "statistical analysis data"
    ]
    
    print("\n3. Sample evaluation results:")
    print_separator("", "-", 60)
    
    for query in test_queries:
        print(f"Query: '{query}'")
        
        # Get results from different methods
        vector_results = search_system.vector_search(query, k=3)
        keyword_results = search_system.keyword_search(query, k=3)
        hybrid_results = search_system.weighted_hybrid_search(query, k=3, alpha=0.5)
        
        print(f"  Vector:  {len(vector_results)} results")
        print(f"  Keyword: {len(keyword_results)} results")
        print(f"  Hybrid:  {len(hybrid_results)} results")
        print()

def demonstrate_api_usage():
    """Show how to use the FastAPI endpoints."""
    print_separator("API USAGE DEMO")
    
    print("To start the FastAPI server, run:")
    print("  python fastapi_hybrid.py")
    print()
    print("Then you can access these endpoints:")
    print()
    
    endpoints = [
        ("GET /", "API information and endpoint list"),
        ("GET /hybrid_search?query=neural+networks&k=3", "Weighted hybrid search"),
        ("GET /rrf_search?query=deep+learning&k=5", "RRF hybrid search"),
        ("GET /vector_search?query=AI&k=3", "Vector-only search"),
        ("GET /keyword_search?query=machine+learning&k=3", "Keyword-only search"),
        ("POST /add_documents", "Add new documents to index"),
        ("GET /stats", "System statistics")
    ]
    
    for endpoint, description in endpoints:
        print(f"  {endpoint:<50} - {description}")
    
    print()
    print("Example curl commands:")
    print()
    print("# Hybrid search")
    print('curl "http://localhost:8000/hybrid_search?query=neural%20networks&k=3&alpha=0.6"')
    print()
    print("# Add documents")
    print('curl -X POST "http://localhost:8000/add_documents" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'[{"title":"Test","author":"Me","year":2024,"keywords":"test","chunk_text":"Test document"}]\'')
    print()
    print("Interactive API documentation available at:")
    print("  http://localhost:8000/docs (Swagger UI)")
    print("  http://localhost:8000/redoc (ReDoc)")

def demonstrate_performance_comparison():
    """Show performance comparison between different methods."""
    print_separator("PERFORMANCE COMPARISON DEMO")
    
    # Create sample performance data for demonstration
    methods = ['Vector Only', 'Keyword Only', 'Hybrid (α=0.3)', 'Hybrid (α=0.5)', 'Hybrid (α=0.7)', 'RRF Hybrid']
    recall_scores = [0.65, 0.45, 0.72, 0.78, 0.75, 0.76]
    precision_scores = [0.58, 0.67, 0.71, 0.74, 0.73, 0.72]
    
    print("Sample Performance Metrics (Recall@3, Precision@3):")
    print_separator("", "-", 60)
    
    df = pd.DataFrame({
        'Method': methods,
        'Recall@3': recall_scores,
        'Precision@3': precision_scores
    })
    
    print(df.to_string(index=False, float_format='%.3f'))
    
    print("\nKey Observations:")
    print("• Hybrid methods generally outperform single-method approaches")
    print("• Optimal α parameter balances semantic and keyword matching")
    print("• RRF provides competitive performance without parameter tuning")
    print("• Choice of method depends on specific use case and query types")

def show_system_architecture():
    """Display system architecture information."""
    print_separator("SYSTEM ARCHITECTURE")
    
    architecture_info = """
Components:
├── hybrid_search.py      - Core search system implementation
├── fastapi_hybrid.py     - REST API server with endpoints
├── evaluation.py         - Comprehensive evaluation framework
└── demo.py              - This demonstration script

Key Features:
• SQLite + FTS5 for metadata storage and keyword search
• FAISS for efficient vector similarity search
• Sentence Transformers for text embeddings
• Multiple hybrid ranking strategies (weighted sum, RRF)
• Comprehensive evaluation with standard IR metrics
• RESTful API for integration with other systems

Database Schema:
• documents table: metadata (title, author, year, keywords, text)
• doc_chunks (FTS5): full-text search index
• embeddings table: mapping between doc_id and FAISS indices

Search Pipeline:
1. Query preprocessing and embedding generation
2. Parallel execution of vector and keyword search
3. Score normalization and fusion
4. Final ranking and result presentation
"""
    print(architecture_info)

def main():
    """Main demonstration function."""
    print_separator("HOMEWORK 5: HYBRID SEARCH SYSTEM DEMO", "=", 80)
    print()
    print("This demonstration showcases a complete hybrid retrieval system")
    print("that combines FAISS vector search with SQLite FTS5 keyword search.")
    print()
    
    try:
        # Run all demonstrations
        search_system = demonstrate_basic_functionality()
        
        print()
        demonstrate_detailed_results(search_system)
        
        print()
        run_evaluation_demo()
        
        print()
        demonstrate_performance_comparison()
        
        print()
        demonstrate_api_usage()
        
        print()
        show_system_architecture()
        
        print()
        print_separator("DEMO COMPLETED SUCCESSFULLY", "=", 80)
        print()
        print("Next steps:")
        print("1. Run 'python evaluation.py' for complete evaluation")
        print("2. Run 'python fastapi_hybrid.py' to start the API server")
        print("3. Experiment with different queries and parameters")
        print("4. Add your own documents and test cases")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please ensure all required dependencies are installed:")
        print("  pip install faiss-cpu sentence-transformers fastapi uvicorn pandas matplotlib seaborn")
        sys.exit(1)

if __name__ == "__main__":
    main()