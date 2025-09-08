import sqlite3
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from rank_bm25 import BM25Okapi
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "hybrid_search.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with documents and FTS5 tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table for metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                year INTEGER,
                keywords TEXT,
                chunk_text TEXT
            )
        ''')
        
        # Create FTS5 virtual table for full-text search
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
                content,
                content='documents',
                content_rowid='doc_id'
            )
        ''')
        
        # Create embeddings table to track FAISS indices
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id INTEGER PRIMARY KEY,
                faiss_idx INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def add_documents(self, documents: List[Dict]) -> List[int]:
        """Add documents with metadata to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        doc_ids = []
        
        for doc in documents:
            # Insert into documents table
            cursor.execute('''
                INSERT INTO documents (title, author, year, keywords, chunk_text)
                VALUES (?, ?, ?, ?, ?)
            ''', (doc['title'], doc['author'], doc['year'], 
                 doc['keywords'], doc['chunk_text']))
            
            doc_id = cursor.lastrowid
            doc_ids.append(doc_id)
            
            # Insert into FTS5 table
            cursor.execute('''
                INSERT INTO doc_chunks(rowid, content) VALUES (?, ?)
            ''', (doc_id, doc['chunk_text']))
        
        conn.commit()
        conn.close()
        logger.info(f"Added {len(documents)} documents to database")
        return doc_ids
    
    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Perform keyword search using SQLite FTS5."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Escape FTS5 special characters
        escaped_query = query.replace('"', '""')
        
        try:
            cursor.execute('''
                SELECT documents.doc_id, rank
                FROM documents
                JOIN doc_chunks ON documents.doc_id = doc_chunks.rowid
                WHERE doc_chunks MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (escaped_query, k))
            
            results = [(doc_id, -rank) for doc_id, rank in cursor.fetchall()]  # Negative rank for descending order
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS query failed: {e}. Falling back to LIKE search.")
            # Fallback to simple LIKE search
            cursor.execute('''
                SELECT doc_id, 1.0 as score
                FROM documents
                WHERE chunk_text LIKE ?
                LIMIT ?
            ''', (f'%{query}%', k))
            
            results = [(doc_id, score) for doc_id, score in cursor.fetchall()]
        
        conn.close()
        return results

    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Perform BM25 keyword search on all documents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all documents with their text
        cursor.execute('''
            SELECT doc_id, chunk_text
            FROM documents
            ORDER BY doc_id
        ''')
        
        documents_data = cursor.fetchall()
        conn.close()
        
        if not documents_data:
            return []
        
        # Prepare documents for BM25
        doc_ids = [doc_id for doc_id, _ in documents_data]
        doc_texts = [text for _, text in documents_data]
        
        # Simple tokenization (split on whitespace and punctuation)
        tokenized_docs = [self._tokenize_text(text) for text in doc_texts]
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_docs)
        
        # Tokenize query
        tokenized_query = self._tokenize_text(query)
        
        # Get BM25 scores for all documents
        scores = bm25.get_scores(tokenized_query)
        
        # Combine doc_ids with scores and sort by score (descending)
        doc_scores = [(doc_ids[i], scores[i]) for i in range(len(doc_ids))]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return doc_scores[:k]

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on whitespace and punctuation."""
        # Convert to lowercase and split on whitespace and common punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def combined_keyword_search(self, query: str, k: int = 10, fts5_weight: float = 0.5, bm25_weight: float = 0.5) -> List[Tuple[int, float]]:
        """Combine FTS5 and BM25 search results with weighted scoring."""
        # Get results from both methods
        fts5_results = dict(self.keyword_search(query, k * 2))
        bm25_results = dict(self.bm25_search(query, k * 2))
        
        # Get all unique document IDs
        all_doc_ids = set(fts5_results.keys()) | set(bm25_results.keys())
        
        if not all_doc_ids:
            return []
        
        # Normalize scores to [0, 1] range for each method
        def normalize_scores(score_dict):
            if not score_dict:
                return {}
            scores = list(score_dict.values())
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return {doc_id: 1.0 for doc_id in score_dict}
            return {doc_id: (score - min_score) / (max_score - min_score) 
                   for doc_id, score in score_dict.items()}
        
        normalized_fts5 = normalize_scores(fts5_results)
        normalized_bm25 = normalize_scores(bm25_results)
        
        # Combine scores with weights
        combined_results = []
        for doc_id in all_doc_ids:
            fts5_score = normalized_fts5.get(doc_id, 0.0)
            bm25_score = normalized_bm25.get(doc_id, 0.0)
            combined_score = fts5_weight * fts5_score + bm25_weight * bm25_score
            combined_results.append((doc_id, combined_score))
        
        # Sort by combined score and return top-k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:k]
    
    def get_document_details(self, doc_ids: List[int]) -> List[Dict]:
        """Retrieve document details by IDs."""
        if not doc_ids:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(doc_ids))
        cursor.execute(f'''
            SELECT doc_id, title, author, year, keywords, chunk_text
            FROM documents
            WHERE doc_id IN ({placeholders})
        ''', doc_ids)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'doc_id': row[0],
                'title': row[1],
                'author': row[2],
                'year': row[3],
                'keywords': row[4],
                'chunk_text': row[5]
            })
        
        conn.close()
        return results
    
    def store_embedding_mapping(self, doc_id: int, faiss_idx: int):
        """Store mapping between document ID and FAISS index."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings (doc_id, faiss_idx) VALUES (?, ?)
        ''', (doc_id, faiss_idx))
        
        conn.commit()
        conn.close()
    
    def get_embedding_mappings(self) -> Dict[int, int]:
        """Get all document ID to FAISS index mappings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT doc_id, faiss_idx FROM embeddings')
        mappings = {doc_id: faiss_idx for doc_id, faiss_idx in cursor.fetchall()}
        
        conn.close()
        return mappings
    
    def get_document_count(self) -> int:
        """Get total number of documents in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def get_embedding_count(self) -> int:
        """Get total number of embeddings in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT doc_id, title, author, year, keywords, chunk_text
            FROM documents
            ORDER BY doc_id
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'doc_id': row[0],
                'title': row[1],
                'author': row[2],
                'year': row[3],
                'keywords': row[4],
                'chunk_text': row[5]
            })
        
        conn.close()
        return results
    
    def clear_database(self):
        """Clear all data from database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM embeddings")
        cursor.execute("DELETE FROM doc_chunks")
        cursor.execute("DELETE FROM documents")
        
        conn.commit()
        conn.close()
        logger.info("Database cleared successfully")
    
    def close(self):
        """Close database connection (placeholder for future connection pooling)."""
        pass