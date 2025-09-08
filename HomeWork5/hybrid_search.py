import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from database_manager import DatabaseManager
from embedding_manager import EmbeddingManager
from faiss_manager import FAISSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridSearchSystem:
    def __init__(self, db_path: str = "hybrid_search.db", model_name: str = "all-MiniLM-L6-v2"):
        # Initialize managers
        self.db_manager = DatabaseManager(db_path)
        self.embedding_manager = EmbeddingManager(model_name)
        self.faiss_manager = FAISSManager(self.embedding_manager.get_dimension())
        
        # Legacy compatibility
        self.db_path = db_path
        self.model = self.embedding_manager.model
        self.dimension = self.embedding_manager.get_dimension()
        self.faiss_index = self.faiss_manager.index
        self.doc_id_to_faiss_idx = self.faiss_manager.doc_id_to_faiss_idx
        self.faiss_idx_to_doc_id = self.faiss_manager.faiss_idx_to_doc_id
    
    def add_documents(self, documents: List[Dict]):
        """Add documents with metadata to the system."""
        # Add documents to database
        doc_ids = self.db_manager.add_documents(documents)
        
        # Generate embeddings
        embeddings = self.embedding_manager.encode_documents(documents)
        
        # Add to FAISS index
        success = self.faiss_manager.add_embeddings(embeddings, doc_ids)
        
        # Store FAISS mappings in database
        faiss_mappings = self.faiss_manager.doc_id_to_faiss_idx
        for doc_id in doc_ids:
            if doc_id in faiss_mappings:
                self.db_manager.store_embedding_mapping(doc_id, faiss_mappings[doc_id])
        
        # Update legacy attributes for compatibility
        self.faiss_index = self.faiss_manager.index
        self.doc_id_to_faiss_idx = self.faiss_manager.doc_id_to_faiss_idx
        self.faiss_idx_to_doc_id = self.faiss_manager.faiss_idx_to_doc_id
        
        logger.info(f"Added {len(documents)} documents to the system")
    
    def vector_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Perform semantic search using FAISS."""
        # Generate query embedding
        query_embedding = self.embedding_manager.encode_query(query)
        
        # Search using FAISS manager
        results = self.faiss_manager.search(query_embedding, k)
        
        return results
    
    def keyword_search(self, query: str, k: int = 10, method: str = "combined") -> List[Tuple[int, float]]:
        """Perform keyword search using FTS5, BM25, or combined approach."""
        if method == "fts5":
            return self.db_manager.keyword_search(query, k)
        elif method == "bm25":
            return self.db_manager.bm25_search(query, k)
        elif method == "combined":
            return self.db_manager.combined_keyword_search(query, k)
        else:
            raise ValueError(f"Unknown keyword search method: {method}. Use 'fts5', 'bm25', or 'combined'.")

    def fts5_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Perform keyword search using SQLite FTS5 only."""
        return self.db_manager.keyword_search(query, k)

    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Perform keyword search using BM25 only."""
        return self.db_manager.bm25_search(query, k)

    def combined_keyword_search(self, query: str, k: int = 10, fts5_weight: float = 0.6, bm25_weight: float = 0.4) -> List[Tuple[int, float]]:
        """Perform combined FTS5 + BM25 keyword search."""
        return self.db_manager.combined_keyword_search(query, k, fts5_weight, bm25_weight)
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def weighted_hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Tuple[int, float]]:
        """Combine vector and keyword search using weighted sum."""
        vector_results = dict(self.vector_search(query, k * 2))
        keyword_results = dict(self.keyword_search(query, k * 2))
        
        # Get all unique document IDs
        all_doc_ids = set(vector_results.keys()) | set(keyword_results.keys())
        
        # Normalize scores
        if vector_results:
            vec_scores = list(vector_results.values())
            normalized_vec_scores = self.normalize_scores(vec_scores)
            vector_results = dict(zip(vector_results.keys(), normalized_vec_scores))
        
        if keyword_results:
            key_scores = list(keyword_results.values())
            normalized_key_scores = self.normalize_scores(key_scores)
            keyword_results = dict(zip(keyword_results.keys(), normalized_key_scores))
        
        # Combine scores
        combined_results = []
        for doc_id in all_doc_ids:
            vec_score = vector_results.get(doc_id, 0.0)
            key_score = keyword_results.get(doc_id, 0.0)
            hybrid_score = alpha * vec_score + (1 - alpha) * key_score
            combined_results.append((doc_id, hybrid_score))
        
        # Sort by score and return top-k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:k]
    
    def rrf_hybrid_search(self, query: str, k: int = 10, rrf_k: int = 60) -> List[Tuple[int, float]]:
        """Combine vector and keyword search using Reciprocal Rank Fusion."""
        vector_results = self.vector_search(query, k * 2)
        keyword_results = self.keyword_search(query, k * 2)
        
        # Create rank dictionaries
        vector_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(vector_results)}
        keyword_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(keyword_results)}
        
        # Get all unique document IDs
        all_doc_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())
        
        # Calculate RRF scores
        rrf_results = []
        for doc_id in all_doc_ids:
            vec_rank = vector_ranks.get(doc_id, float('inf'))
            key_rank = keyword_ranks.get(doc_id, float('inf'))
            
            rrf_score = 0
            if vec_rank != float('inf'):
                rrf_score += 1 / (rrf_k + vec_rank)
            if key_rank != float('inf'):
                rrf_score += 1 / (rrf_k + key_rank)
            
            rrf_results.append((doc_id, rrf_score))
        
        # Sort by RRF score and return top-k
        rrf_results.sort(key=lambda x: x[1], reverse=True)
        return rrf_results[:k]
    
    def get_document_details(self, doc_ids: List[int]) -> List[Dict]:
        """Retrieve document details by IDs."""
        return self.db_manager.get_document_details(doc_ids)
    
    def load_faiss_mappings(self):
        """Load FAISS index mappings from database."""
        mappings = self.db_manager.get_embedding_mappings()
        
        # Update FAISS manager mappings
        for doc_id, faiss_idx in mappings.items():
            self.faiss_manager.doc_id_to_faiss_idx[doc_id] = faiss_idx
            self.faiss_manager.faiss_idx_to_doc_id[faiss_idx] = doc_id
        
        # Update legacy attributes for compatibility
        self.doc_id_to_faiss_idx = self.faiss_manager.doc_id_to_faiss_idx
        self.faiss_idx_to_doc_id = self.faiss_manager.faiss_idx_to_doc_id
    
    def save_index(self, faiss_path: str = "faiss_index.bin"):
        """Save FAISS index to disk."""
        self.faiss_manager.save_index(faiss_path)
    
    def load_index(self, faiss_path: str = "faiss_index.bin"):
        """Load FAISS index from disk."""
        success = self.faiss_manager.load_index(faiss_path)
        if success:
            self.load_faiss_mappings()
            # Update legacy attributes for compatibility
            self.faiss_index = self.faiss_manager.index
            self.doc_id_to_faiss_idx = self.faiss_manager.doc_id_to_faiss_idx
            self.faiss_idx_to_doc_id = self.faiss_manager.faiss_idx_to_doc_id
            logger.info(f"FAISS index loaded from {faiss_path}")
        else:
            logger.warning(f"Failed to load FAISS index from {faiss_path}")
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics."""
        return {
            'database_stats': {
                'total_documents': self.db_manager.get_document_count(),
                'total_embeddings': self.db_manager.get_embedding_count()
            },
            'embedding_stats': self.embedding_manager.get_model_info(),
            'faiss_stats': self.faiss_manager.get_index_stats()
        }


# Sample usage
if __name__ == "__main__":
    # Initialize the system
    search_system = HybridSearchSystem()
    
    # Sample documents
    sample_docs = [
        {
            'title': 'Introduction to Machine Learning',
            'author': 'John Smith',
            'year': 2023,
            'keywords': 'machine learning, artificial intelligence, neural networks',
            'chunk_text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.'
        },
        {
            'title': 'Deep Learning Fundamentals',
            'author': 'Jane Doe',
            'year': 2022,
            'keywords': 'deep learning, neural networks, backpropagation',
            'chunk_text': 'Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.'
        },
        {
            'title': 'Natural Language Processing',
            'author': 'Bob Johnson',
            'year': 2024,
            'keywords': 'NLP, text processing, language models',
            'chunk_text': 'Natural language processing enables computers to understand, interpret, and generate human language in a meaningful way.'
        }
    ]
    
    # Add documents
    search_system.add_documents(sample_docs)
    
    # Test different search methods
    query = "neural networks deep learning"
    print(f"Query: {query}\n")
    
    # Vector search
    vector_results = search_system.vector_search(query, k=3)
    print("Vector Search Results:")
    for doc_id, score in vector_results:
        docs = search_system.get_document_details([doc_id])
        if docs:
            print(f"  Doc {doc_id}: {docs[0]['title']} (Score: {score:.4f})")
    
    # FTS5 search
    fts5_results = search_system.fts5_search(query, k=3)
    print("\nFTS5 Search Results:")
    for doc_id, score in fts5_results:
        docs = search_system.get_document_details([doc_id])
        if docs:
            print(f"  Doc {doc_id}: {docs[0]['title']} (Score: {score:.4f})")

    # BM25 search
    bm25_results = search_system.bm25_search(query, k=3)
    print("\nBM25 Search Results:")
    for doc_id, score in bm25_results:
        docs = search_system.get_document_details([doc_id])
        if docs:
            print(f"  Doc {doc_id}: {docs[0]['title']} (Score: {score:.4f})")

    # Combined keyword search (FTS5 + BM25)
    combined_kw_results = search_system.combined_keyword_search(query, k=3)
    print("\nCombined Keyword Search Results (FTS5 + BM25):")
    for doc_id, score in combined_kw_results:
        docs = search_system.get_document_details([doc_id])
        if docs:
            print(f"  Doc {doc_id}: {docs[0]['title']} (Score: {score:.4f})")

    # Default keyword search (now uses combined method)
    keyword_results = search_system.keyword_search(query, k=3)
    print("\nDefault Keyword Search Results (Combined):")
    for doc_id, score in keyword_results:
        docs = search_system.get_document_details([doc_id])
        if docs:
            print(f"  Doc {doc_id}: {docs[0]['title']} (Score: {score:.4f})")
    
    # Weighted hybrid search
    hybrid_results = search_system.weighted_hybrid_search(query, k=3, alpha=0.6)
    print("\nWeighted Hybrid Search Results:")
    for doc_id, score in hybrid_results:
        docs = search_system.get_document_details([doc_id])
        if docs:
            print(f"  Doc {doc_id}: {docs[0]['title']} (Score: {score:.4f})")
    
    # RRF hybrid search
    rrf_results = search_system.rrf_hybrid_search(query, k=3)
    print("\nRRF Hybrid Search Results:")
    for doc_id, score in rrf_results:
        docs = search_system.get_document_details([doc_id])
        if docs:
            print(f"  Doc {doc_id}: {docs[0]['title']} (Score: {score:.4f})")
    
    # Print system stats
    print("\nSystem Statistics:")
    stats = search_system.get_system_stats()
    print(f"  Documents: {stats['database_stats']['total_documents']}")
    print(f"  Embeddings: {stats['database_stats']['total_embeddings']}")
    print(f"  FAISS vectors: {stats['faiss_stats']['total_vectors']}")
    
    # Save the index
    search_system.save_index()