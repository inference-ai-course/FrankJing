import numpy as np
import logging
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_name} (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.model:
            return {
                'model_name': self.model_name,
                'dimension': self.dimension,
                'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown')
            }
        return {}
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into embeddings."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Handle single string or list of strings
        if isinstance(text, str):
            text = [text]
        
        # Generate embeddings
        embeddings = self.model.encode(text, convert_to_numpy=True)
        
        # Ensure float32 for FAISS compatibility
        return embeddings.astype('float32')
    
    def encode_documents(self, documents: List[Dict]) -> np.ndarray:
        """Encode document texts into embeddings."""
        texts = [doc['chunk_text'] for doc in documents]
        return self.encode_text(texts)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into embedding."""
        return self.encode_text(query)[0]  # Return single embedding, not array
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text before encoding."""
        # Basic text cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s\-.,;:!?()]', '', text)  # Remove special chars except basic punctuation
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        # Preprocess the text
        text = self.preprocess_text(text)
        
        # Simple word-based chunking
        words = text.split()
        
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            if end >= len(words):
                break
                
            start = end - overlap
        
        return chunks
    
    def create_document_chunks(self, documents: List[Dict], chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """Create chunked documents from original documents."""
        chunked_docs = []
        
        for doc in documents:
            text = doc['chunk_text']
            chunks = self.chunk_text(text, chunk_size, overlap)
            
            for i, chunk in enumerate(chunks):
                chunked_doc = doc.copy()
                chunked_doc['chunk_text'] = chunk
                chunked_doc['chunk_id'] = f"{doc.get('doc_id', 'unknown')}_{i}"
                chunked_doc['original_doc_id'] = doc.get('doc_id')
                chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches for memory efficiency."""
        if not texts:
            return np.array([]).reshape(0, self.dimension).astype('float32')
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode_text(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def find_similar_embeddings(self, query_embedding: np.ndarray, 
                               document_embeddings: np.ndarray, 
                               top_k: int = 10) -> List[tuple]:
        """Find most similar embeddings using cosine similarity."""
        if document_embeddings.shape[0] == 0:
            return []
        
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(doc_norms, query_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return (index, similarity) pairs
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to disk."""
        np.save(filepath, embeddings)
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> Optional[np.ndarray]:
        """Load embeddings from disk."""
        try:
            if Path(filepath).exists():
                embeddings = np.load(filepath)
                logger.info(f"Embeddings loaded from {filepath}")
                return embeddings.astype('float32')
            else:
                logger.warning(f"Embeddings file {filepath} not found")
                return None
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def get_text_stats(self, texts: List[str]) -> Dict:
        """Get statistics about text collection."""
        if not texts:
            return {}
        
        lengths = [len(text.split()) for text in texts]
        
        return {
            'total_texts': len(texts),
            'total_words': sum(lengths),
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'std_length': np.std(lengths)
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Test documents
    test_docs = [
        {
            'title': 'Machine Learning Basics',
            'chunk_text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.'
        },
        {
            'title': 'Deep Learning Overview',
            'chunk_text': 'Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data.'
        }
    ]
    
    # Test embedding generation
    embeddings = embedding_manager.encode_documents(test_docs)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test query encoding
    query = "artificial intelligence neural networks"
    query_embedding = embedding_manager.encode_query(query)
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Test similarity calculation
    similarity = embedding_manager.calculate_similarity(query_embedding, embeddings[0])
    print(f"Similarity between query and first document: {similarity:.4f}")
    
    # Test text chunking
    long_text = " ".join([test_docs[0]['chunk_text']] * 10)  # Create long text
    chunks = embedding_manager.chunk_text(long_text, chunk_size=20, overlap=5)
    print(f"Created {len(chunks)} chunks from long text")
    
    print("Embedding manager test completed successfully!")