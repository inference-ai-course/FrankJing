#!/usr/bin/env python3
"""
Week 5: Hybrid Retrieval System
===============================

This module implements a hybrid retrieval system that combines:
1. Dense vector search (FAISS) for semantic similarity
2. Sparse keyword search (SQLite FTS5) for exact matches
3. Score fusion strategies (weighted sum and reciprocal rank fusion)

The system extends the Week 4 RAG pipeline by adding metadata storage
and keyword search capabilities.
"""

import sqlite3
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
import argparse
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with metadata and scores."""
    chunk_id: str
    text: str
    source_name: str
    source_path: str
    start_token: int
    end_token: int
    vector_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0
    rank: int = 0

class HybridRetrievalSystem:
    """Hybrid retrieval system combining FAISS and SQLite FTS5."""
    
    def __init__(self, db_path: str = "hybrid_index.db", faiss_path: str = "../HomeWork4/faiss.index"):
        """Initialize the hybrid retrieval system.
        
        Args:
            db_path: Path to SQLite database
            faiss_path: Path to existing FAISS index from Week 4
        """
        self.db_path = db_path
        self.faiss_path = faiss_path
        self.model = None
        self.faiss_index = None
        self.embeddings = None
        self.metadata = None
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with schema for documents and FTS5 chunks."""
        conn = sqlite3.connect(self.db_path)
        
        # Create documents table for metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                year INTEGER,
                keywords TEXT,
                source_path TEXT,
                source_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create FTS5 virtual table for full-text search on chunks
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
                chunk_id,
                content,
                source_name,
                content='documents',
                content_rowid='doc_id'
            )
        """)
        
        # Create chunks table for storing chunk metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id INTEGER,
                start_token INTEGER,
                end_token INTEGER,
                text TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def load_existing_index(self, chunks_path: str = "../HomeWork4/chunks.jsonl", 
                           embeddings_path: str = "../HomeWork4/embeddings.npy",
                           meta_path: str = "../HomeWork4/meta.json"):
        """Load existing FAISS index and embeddings from Week 4."""
        try:
            # Load metadata
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Load embeddings
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings: {self.embeddings.shape}")
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(self.faiss_path)
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            
            # Load sentence transformer model
            model_name = self.metadata.get('model', 'all-MiniLM-L6-v2')
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded model: {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load existing index: {e}")
            return False
    
    def build_hybrid_index(self, chunks_path: str = "../HomeWork4/chunks.jsonl"):
        """Build hybrid index by importing existing chunks into SQLite."""
        if not self.load_existing_index():
            logger.error("Failed to load existing index")
            return False
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Clear existing data
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM doc_chunks")
            conn.execute("DELETE FROM documents")
            
            # Process chunks
            doc_counter = 0
            chunk_counter = 0
            
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk_data = json.loads(line.strip())
                    
                    # Extract document info
                    source_name = chunk_data['source_name']
                    source_path = chunk_data['source_path']
                    
                    # Check if document already exists
                    cursor = conn.execute(
                        "SELECT doc_id FROM documents WHERE source_name = ?", 
                        (source_name,)
                    )
                    doc_row = cursor.fetchone()
                    
                    if doc_row is None:
                        # Create new document entry
                        doc_counter += 1
                        conn.execute("""
                            INSERT INTO documents (doc_id, title, author, year, keywords, source_path, source_name)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            doc_counter,
                            source_name.replace('.pdf', ''),  # Use filename as title
                            "Unknown",  # We don't have author info
                            2024,  # Default year
                            "",  # No keywords extracted
                            source_path,
                            source_name
                        ))
                        doc_id = doc_counter
                    else:
                        doc_id = doc_row[0]
                    
                    # Insert chunk
                    chunk_id = chunk_data['chunk_id']
                    conn.execute("""
                        INSERT INTO chunks (chunk_id, doc_id, start_token, end_token, text)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        chunk_id,
                        doc_id,
                        chunk_data['start_token'],
                        chunk_data['end_token'],
                        chunk_data['text']
                    ))
                    
                    # Insert into FTS5 table
                    conn.execute("""
                        INSERT INTO doc_chunks (rowid, chunk_id, content, source_name)
                        VALUES (?, ?, ?, ?)
                    """, (
                        doc_id,
                        chunk_id,
                        chunk_data['text'],
                        source_name
                    ))
                    
                    chunk_counter += 1
            
            conn.commit()
            logger.info(f"Built hybrid index: {doc_counter} documents, {chunk_counter} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build hybrid index: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def vector_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Perform semantic search using FAISS."""
        if not self.model or not self.faiss_index:
            raise ValueError("Model or FAISS index not loaded")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search FAISS
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Get chunk data from database
        conn = sqlite3.connect(self.db_path)
        results = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
                
            cursor = conn.execute("""
                SELECT c.chunk_id, c.text, c.start_token, c.end_token, 
                       d.source_name, d.source_path
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE c.chunk_id = (
                    SELECT chunk_id FROM chunks LIMIT 1 OFFSET ?
                )
            """, (idx,))
            
            row = cursor.fetchone()
            if row:
                result = SearchResult(
                    chunk_id=row[0],
                    text=row[1],
                    source_name=row[4],
                    source_path=row[5],
                    start_token=row[2],
                    end_token=row[3],
                    vector_score=1.0 / (1.0 + distance),  # Convert distance to similarity
                    rank=i + 1
                )
                results.append(result)
        
        conn.close()
        return results
    
    def keyword_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Perform keyword search using SQLite FTS5."""
        conn = sqlite3.connect(self.db_path)
        
        # FTS5 search with ranking
        cursor = conn.execute("""
            SELECT c.chunk_id, c.text, c.start_token, c.end_token,
                   d.source_name, d.source_path,
                   rank
            FROM doc_chunks
            JOIN chunks c ON doc_chunks.chunk_id = c.chunk_id
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE doc_chunks MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, k))
        
        results = []
        for i, row in enumerate(cursor):
            # FTS5 rank is lower for better matches, convert to similarity score
            keyword_score = 1.0 / (1.0 + row[6]) if row[6] > 0 else 1.0
            
            result = SearchResult(
                chunk_id=row[0],
                text=row[1],
                source_name=row[4],
                source_path=row[5],
                start_token=row[2],
                end_token=row[3],
                keyword_score=keyword_score,
                rank=i + 1
            )
            results.append(result)
        
        conn.close()
        return results
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.6, 
                     fusion_method: str = "weighted") -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword results.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for vector search (1-alpha for keyword search)
            fusion_method: "weighted" or "rrf" (reciprocal rank fusion)
        """
        # Get results from both methods
        vector_results = self.vector_search(query, k * 2)  # Get more for fusion
        keyword_results = self.keyword_search(query, k * 2)
        
        if fusion_method == "weighted":
            return self._weighted_fusion(vector_results, keyword_results, k, alpha)
        elif fusion_method == "rrf":
            return self._reciprocal_rank_fusion(vector_results, keyword_results, k)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def _weighted_fusion(self, vector_results: List[SearchResult], 
                        keyword_results: List[SearchResult], 
                        k: int, alpha: float) -> List[SearchResult]:
        """Combine results using weighted score fusion."""
        # Create a dictionary to store combined results
        combined = {}
        
        # Add vector results
        for result in vector_results:
            combined[result.chunk_id] = result
            result.hybrid_score = alpha * result.vector_score
        
        # Add keyword results and combine scores
        for result in keyword_results:
            if result.chunk_id in combined:
                # Combine scores
                combined[result.chunk_id].keyword_score = result.keyword_score
                combined[result.chunk_id].hybrid_score += (1 - alpha) * result.keyword_score
            else:
                # New result from keyword search
                result.hybrid_score = (1 - alpha) * result.keyword_score
                combined[result.chunk_id] = result
        
        # Sort by hybrid score and return top k
        sorted_results = sorted(combined.values(), key=lambda x: x.hybrid_score, reverse=True)
        return sorted_results[:k]
    
    def _reciprocal_rank_fusion(self, vector_results: List[SearchResult], 
                               keyword_results: List[SearchResult], 
                               k: int) -> List[SearchResult]:
        """Combine results using reciprocal rank fusion."""
        combined = {}
        
        # Add vector results with RRF scores
        for i, result in enumerate(vector_results):
            rrf_score = 1.0 / (60 + i + 1)  # RRF with k=60
            if result.chunk_id in combined:
                combined[result.chunk_id].hybrid_score += rrf_score
            else:
                result.hybrid_score = rrf_score
                combined[result.chunk_id] = result
        
        # Add keyword results with RRF scores
        for i, result in enumerate(keyword_results):
            rrf_score = 1.0 / (60 + i + 1)
            if result.chunk_id in combined:
                combined[result.chunk_id].hybrid_score += rrf_score
            else:
                result.hybrid_score = rrf_score
                combined[result.chunk_id] = result
        
        # Sort by hybrid score and return top k
        sorted_results = sorted(combined.values(), key=lambda x: x.hybrid_score, reverse=True)
        return sorted_results[:k]
    
    def evaluate_search(self, test_queries: List[Dict[str, any]], k: int = 3) -> Dict[str, float]:
        """Evaluate search performance on test queries.
        
        Args:
            test_queries: List of dicts with 'query' and 'relevant_chunks' keys
            k: Number of top results to consider for evaluation
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'vector_hit_rate': 0.0,
            'keyword_hit_rate': 0.0,
            'hybrid_hit_rate': 0.0,
            'vector_recall': 0.0,
            'keyword_recall': 0.0,
            'hybrid_recall': 0.0
        }
        
        total_queries = len(test_queries)
        
        for query_data in test_queries:
            query = query_data['query']
            relevant_chunks = set(query_data['relevant_chunks'])
            
            # Get results from each method
            vector_results = self.vector_search(query, k)
            keyword_results = self.keyword_search(query, k)
            hybrid_results = self.hybrid_search(query, k)
            
            # Calculate hit rates (any relevant chunk in top-k)
            vector_hit = any(result.chunk_id in relevant_chunks for result in vector_results)
            keyword_hit = any(result.chunk_id in relevant_chunks for result in keyword_results)
            hybrid_hit = any(result.chunk_id in relevant_chunks for result in hybrid_results)
            
            metrics['vector_hit_rate'] += vector_hit
            metrics['keyword_hit_rate'] += keyword_hit
            metrics['hybrid_hit_rate'] += hybrid_hit
            
            # Calculate recall (proportion of relevant chunks found)
            vector_recall = len([r for r in vector_results if r.chunk_id in relevant_chunks]) / len(relevant_chunks)
            keyword_recall = len([r for r in keyword_results if r.chunk_id in relevant_chunks]) / len(relevant_chunks)
            hybrid_recall = len([r for r in hybrid_results if r.chunk_id in relevant_chunks]) / len(relevant_chunks)
            
            metrics['vector_recall'] += vector_recall
            metrics['keyword_recall'] += keyword_recall
            metrics['hybrid_recall'] += hybrid_recall
        
        # Average over all queries
        for key in metrics:
            metrics[key] /= total_queries
        
        return metrics

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Hybrid Retrieval System")
    parser.add_argument("command", choices=["build", "search", "evaluate"], 
                       help="Command to execute")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--k", type=int, default=3, help="Number of results")
    parser.add_argument("--alpha", type=float, default=0.6, help="Vector search weight")
    parser.add_argument("--fusion", choices=["weighted", "rrf"], default="weighted",
                       help="Fusion method")
    parser.add_argument("--db", default="hybrid_index.db", help="Database path")
    
    args = parser.parse_args()
    
    # Initialize system
    system = HybridRetrievalSystem(db_path=args.db)
    
    if args.command == "build":
        logger.info("Building hybrid index...")
        if system.build_hybrid_index():
            logger.info("Hybrid index built successfully!")
        else:
            logger.error("Failed to build hybrid index")
    
    elif args.command == "search":
        if not args.query:
            logger.error("Query required for search command")
            return
        
        logger.info(f"Searching for: {args.query}")
        
        # Perform searches
        vector_results = system.vector_search(args.query, args.k)
        keyword_results = system.keyword_search(args.query, args.k)
        hybrid_results = system.hybrid_search(args.query, args.k, args.alpha, args.fusion)
        
        # Display results
        print(f"\n=== Vector Search Results (Top {args.k}) ===")
        for i, result in enumerate(vector_results, 1):
            print(f"{i}. {result.chunk_id} (score: {result.vector_score:.3f})")
            print(f"   {result.text[:200]}...")
            print()
        
        print(f"\n=== Keyword Search Results (Top {args.k}) ===")
        for i, result in enumerate(keyword_results, 1):
            print(f"{i}. {result.chunk_id} (score: {result.keyword_score:.3f})")
            print(f"   {result.text[:200]}...")
            print()
        
        print(f"\n=== Hybrid Search Results (Top {args.k}) ===")
        for i, result in enumerate(hybrid_results, 1):
            print(f"{i}. {result.chunk_id} (hybrid: {result.hybrid_score:.3f}, "
                  f"vector: {result.vector_score:.3f}, keyword: {result.keyword_score:.3f})")
            print(f"   {result.text[:200]}...")
            print()
    
    elif args.command == "evaluate":
        # Load test queries (you would define these based on your data)
        test_queries = [
            {
                "query": "machine learning algorithms",
                "relevant_chunks": ["paper_1:0-512", "paper_2:0-512"]  # Example
            },
            {
                "query": "neural networks",
                "relevant_chunks": ["paper_3:0-512", "paper_4:0-512"]  # Example
            }
        ]
        
        logger.info("Evaluating search performance...")
        metrics = system.evaluate_search(test_queries, args.k)
        
        print("\n=== Evaluation Results ===")
        print(f"Hit Rate @{args.k}:")
        print(f"  Vector:   {metrics['vector_hit_rate']:.3f}")
        print(f"  Keyword:  {metrics['keyword_hit_rate']:.3f}")
        print(f"  Hybrid:   {metrics['hybrid_hit_rate']:.3f}")
        print(f"\nRecall @{args.k}:")
        print(f"  Vector:   {metrics['vector_recall']:.3f}")
        print(f"  Keyword:  {metrics['keyword_recall']:.3f}")
        print(f"  Hybrid:   {metrics['hybrid_recall']:.3f}")

if __name__ == "__main__":
    main()
