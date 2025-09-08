import faiss
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSManager:
    def __init__(self, dimension: int, index_type: str = "IndexFlatIP"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.doc_id_to_faiss_idx = {}
        self.faiss_idx_to_doc_id = {}
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on specified type."""
        try:
            if self.index_type == "IndexFlatIP":
                # Inner product (cosine similarity with normalized vectors)
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IndexFlatL2":
                # L2 distance
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IndexIVFFlat":
                # IVF (Inverted File) for faster search with large datasets
                quantizer = faiss.IndexFlatL2(self.dimension)
                nlist = 100  # number of clusters
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
            elif self.index_type == "IndexHNSW":
                # Hierarchical Navigable Small World for approximate search
                M = 16  # number of connections
                self.index = faiss.IndexHNSWFlat(self.dimension, M)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
            else:
                logger.warning(f"Unknown index type {self.index_type}, using IndexFlatIP")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index_type = "IndexFlatIP"
            
            logger.info(f"Created FAISS index: {self.index_type} (dimension: {self.dimension})")
        
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise
    
    def add_embeddings(self, embeddings: np.ndarray, doc_ids: List[int]) -> bool:
        """Add embeddings to FAISS index with document ID mapping."""
        try:
            if embeddings.shape[0] != len(doc_ids):
                raise ValueError("Number of embeddings must match number of document IDs")
            
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")
            
            # Ensure float32 for FAISS
            embeddings = embeddings.astype('float32')
            
            # Normalize vectors for cosine similarity (if using IndexFlatIP)
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(embeddings)
            
            # Train index if necessary (for IVF-based indices)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if embeddings.shape[0] >= 100:  # Need enough data to train
                    logger.info("Training FAISS index...")
                    self.index.train(embeddings)
                else:
                    logger.warning("Not enough data to train IVF index, switching to flat index")
                    self.index = faiss.IndexFlatIP(self.dimension)
            
            # Track mapping between doc_id and FAISS index
            start_idx = self.index.ntotal
            for i, doc_id in enumerate(doc_ids):
                faiss_idx = start_idx + i
                self.doc_id_to_faiss_idx[doc_id] = faiss_idx
                self.faiss_idx_to_doc_id[faiss_idx] = doc_id
            
            # Add to index
            self.index.add(embeddings)
            
            logger.info(f"Added {len(embeddings)} embeddings to FAISS index (total: {self.index.ntotal})")
            return True
        
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Search for similar embeddings in FAISS index."""
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Ensure query is 2D array and float32
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.astype('float32')
            
            # Normalize for cosine similarity (if using IndexFlatIP)
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(query_embedding)
            
            # Perform search
            k = min(k, self.index.ntotal)  # Don't search for more than available
            scores, indices = self.index.search(query_embedding, k)
            
            # Convert results to doc_ids
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # Valid result
                    doc_id = self.faiss_idx_to_doc_id.get(idx)
                    if doc_id is not None:
                        results.append((doc_id, float(score)))
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []
    
    def get_embedding_by_doc_id(self, doc_id: int) -> Optional[np.ndarray]:
        """Get embedding for a specific document ID."""
        try:
            faiss_idx = self.doc_id_to_faiss_idx.get(doc_id)
            if faiss_idx is None:
                return None
            
            # Reconstruct vector from index (if supported)
            if hasattr(self.index, 'reconstruct'):
                embedding = self.index.reconstruct(faiss_idx)
                return embedding.astype('float32')
            else:
                logger.warning("Index type does not support vector reconstruction")
                return None
        
        except Exception as e:
            logger.error(f"Error retrieving embedding for doc_id {doc_id}: {e}")
            return None
    
    def remove_embeddings(self, doc_ids: List[int]) -> bool:
        """Remove embeddings by document IDs (requires index rebuild)."""
        try:
            # For flat indices, we need to rebuild the entire index
            if self.index_type in ["IndexFlatIP", "IndexFlatL2"]:
                logger.info("Rebuilding FAISS index to remove embeddings...")
                
                # Get all current embeddings except those to be removed
                remaining_doc_ids = [doc_id for doc_id in self.doc_id_to_faiss_idx.keys() 
                                   if doc_id not in doc_ids]
                
                if not remaining_doc_ids:
                    # If removing all embeddings, just create new empty index
                    self._create_index()
                    self.doc_id_to_faiss_idx.clear()
                    self.faiss_idx_to_doc_id.clear()
                    return True
                
                # This would require storing original embeddings, which is complex
                # For now, log that this operation is not fully supported
                logger.warning("Full embedding removal requires storing original vectors")
                logger.warning("Consider rebuilding the index from scratch if needed")
                return False
            else:
                logger.warning(f"Removal not supported for index type {self.index_type}")
                return False
        
        except Exception as e:
            logger.error(f"Error removing embeddings: {e}")
            return False
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the FAISS index."""
        stats = {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal if self.index else 0,
            'is_trained': getattr(self.index, 'is_trained', True) if self.index else False,
            'mappings_count': len(self.doc_id_to_faiss_idx)
        }
        
        # Add type-specific stats
        if self.index_type == "IndexIVFFlat" and self.index:
            stats['nlist'] = self.index.nlist
            stats['nprobe'] = self.index.nprobe
        elif self.index_type == "IndexHNSW" and self.index:
            stats['M'] = self.index.hnsw.M
            stats['efConstruction'] = self.index.hnsw.efConstruction
            stats['efSearch'] = self.index.hnsw.efSearch
        
        return stats
    
    def save_index(self, index_path: str, mappings_path: str = None):
        """Save FAISS index and mappings to disk."""
        try:
            if self.index is None:
                logger.warning("No index to save")
                return False
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save mappings
            if mappings_path is None:
                mappings_path = index_path.replace('.bin', '_mappings.pkl')
            
            mappings_data = {
                'doc_id_to_faiss_idx': self.doc_id_to_faiss_idx,
                'faiss_idx_to_doc_id': self.faiss_idx_to_doc_id,
                'index_type': self.index_type,
                'dimension': self.dimension
            }
            
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings_data, f)
            
            logger.info(f"FAISS index saved to {index_path}")
            logger.info(f"Mappings saved to {mappings_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            return False
    
    def load_index(self, index_path: str, mappings_path: str = None) -> bool:
        """Load FAISS index and mappings from disk."""
        try:
            if not Path(index_path).exists():
                logger.warning(f"Index file {index_path} not found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load mappings
            if mappings_path is None:
                mappings_path = index_path.replace('.bin', '_mappings.pkl')
            
            if Path(mappings_path).exists():
                with open(mappings_path, 'rb') as f:
                    mappings_data = pickle.load(f)
                
                self.doc_id_to_faiss_idx = mappings_data.get('doc_id_to_faiss_idx', {})
                self.faiss_idx_to_doc_id = mappings_data.get('faiss_idx_to_doc_id', {})
                self.index_type = mappings_data.get('index_type', self.index_type)
                self.dimension = mappings_data.get('dimension', self.dimension)
            else:
                logger.warning(f"Mappings file {mappings_path} not found")
                self.doc_id_to_faiss_idx.clear()
                self.faiss_idx_to_doc_id.clear()
            
            logger.info(f"FAISS index loaded from {index_path}")
            logger.info(f"Loaded {self.index.ntotal} vectors with {len(self.doc_id_to_faiss_idx)} mappings")
            return True
        
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False
    
    def reset_index(self):
        """Reset the FAISS index to empty state."""
        self._create_index()
        self.doc_id_to_faiss_idx.clear()
        self.faiss_idx_to_doc_id.clear()
        logger.info("FAISS index reset to empty state")
    
    def set_search_parameters(self, **kwargs):
        """Set search parameters for specific index types."""
        try:
            if self.index_type == "IndexIVFFlat":
                if 'nprobe' in kwargs:
                    self.index.nprobe = kwargs['nprobe']
                    logger.info(f"Set nprobe to {kwargs['nprobe']}")
            
            elif self.index_type == "IndexHNSW":
                if 'efSearch' in kwargs:
                    self.index.hnsw.efSearch = kwargs['efSearch']
                    logger.info(f"Set efSearch to {kwargs['efSearch']}")
            
            else:
                logger.info(f"No search parameters available for {self.index_type}")
        
        except Exception as e:
            logger.error(f"Error setting search parameters: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test FAISS manager
    dimension = 384  # Example dimension
    faiss_manager = FAISSManager(dimension)
    
    # Create test embeddings
    num_docs = 100
    test_embeddings = np.random.random((num_docs, dimension)).astype('float32')
    test_doc_ids = list(range(1, num_docs + 1))
    
    # Add embeddings
    success = faiss_manager.add_embeddings(test_embeddings, test_doc_ids)
    print(f"Added embeddings: {success}")
    
    # Test search
    query_embedding = np.random.random((dimension,)).astype('float32')
    results = faiss_manager.search(query_embedding, k=5)
    print(f"Search results: {len(results)} found")
    
    # Print stats
    stats = faiss_manager.get_index_stats()
    print("Index stats:", stats)
    
    # Test save/load
    faiss_manager.save_index("test_index.bin")
    
    # Create new manager and load
    new_manager = FAISSManager(dimension)
    loaded = new_manager.load_index("test_index.bin")
    print(f"Index loaded: {loaded}")
    
    print("FAISS manager test completed successfully!")