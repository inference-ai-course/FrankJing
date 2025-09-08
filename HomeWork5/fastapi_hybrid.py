from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from hybrid_search import HybridSearchSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid Search API",
    description="A hybrid retrieval system combining FAISS vector search with SQLite FTS5 keyword search",
    version="1.0.0"
)

# Initialize the search system
search_system = HybridSearchSystem()

# Load existing index if available
search_system.load_index()

class Document(BaseModel):
    title: str
    author: str
    year: int
    keywords: str
    chunk_text: str

class SearchResult(BaseModel):
    doc_id: int
    title: str
    author: str
    year: int
    keywords: str
    chunk_text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    method: str
    results: List[SearchResult]
    total_results: int

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Hybrid Search API",
        "description": "Combines FAISS vector search with SQLite FTS5 keyword search",
        "endpoints": {
            "/hybrid_search": "Perform hybrid search using weighted combination",
            "/rrf_search": "Perform hybrid search using Reciprocal Rank Fusion",
            "/vector_search": "Perform semantic vector search only",
            "/keyword_search": "Perform keyword search only",
            "/add_documents": "Add new documents to the index",
            "/stats": "Get system statistics"
        }
    }

@app.get("/hybrid_search", response_model=SearchResponse)
async def hybrid_search(
    query: str, 
    k: int = 3, 
    alpha: float = 0.5
):
    """
    Perform hybrid search using weighted combination of vector and keyword search.
    
    - **query**: Search query string
    - **k**: Number of top results to return (default: 3)
    - **alpha**: Weight for vector search (0.0-1.0, default: 0.5)
    """
    try:
        results = search_system.weighted_hybrid_search(query, k, alpha)
        
        if not results:
            return SearchResponse(
                query=query,
                method=f"weighted_hybrid_alpha_{alpha}",
                results=[],
                total_results=0
            )
        
        doc_ids = [doc_id for doc_id, _ in results]
        doc_details = search_system.get_document_details(doc_ids)
        doc_map = {doc['doc_id']: doc for doc in doc_details}
        
        search_results = []
        for doc_id, score in results:
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                search_results.append(SearchResult(
                    doc_id=doc_id,
                    title=doc['title'],
                    author=doc['author'],
                    year=doc['year'],
                    keywords=doc['keywords'],
                    chunk_text=doc['chunk_text'],
                    score=score
                ))
        
        return SearchResponse(
            query=query,
            method=f"weighted_hybrid_alpha_{alpha}",
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rrf_search", response_model=SearchResponse)
async def rrf_search(
    query: str, 
    k: int = 3, 
    rrf_k: int = 60
):
    """
    Perform hybrid search using Reciprocal Rank Fusion.
    
    - **query**: Search query string
    - **k**: Number of top results to return (default: 3)
    - **rrf_k**: RRF parameter (default: 60)
    """
    try:
        results = search_system.rrf_hybrid_search(query, k, rrf_k)
        
        if not results:
            return SearchResponse(
                query=query,
                method=f"rrf_k_{rrf_k}",
                results=[],
                total_results=0
            )
        
        doc_ids = [doc_id for doc_id, _ in results]
        doc_details = search_system.get_document_details(doc_ids)
        doc_map = {doc['doc_id']: doc for doc in doc_details}
        
        search_results = []
        for doc_id, score in results:
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                search_results.append(SearchResult(
                    doc_id=doc_id,
                    title=doc['title'],
                    author=doc['author'],
                    year=doc['year'],
                    keywords=doc['keywords'],
                    chunk_text=doc['chunk_text'],
                    score=score
                ))
        
        return SearchResponse(
            query=query,
            method=f"rrf_k_{rrf_k}",
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Error in RRF search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector_search", response_model=SearchResponse)
async def vector_search(query: str, k: int = 3):
    """
    Perform semantic vector search only using FAISS.
    
    - **query**: Search query string
    - **k**: Number of top results to return (default: 3)
    """
    try:
        results = search_system.vector_search(query, k)
        
        if not results:
            return SearchResponse(
                query=query,
                method="vector_only",
                results=[],
                total_results=0
            )
        
        doc_ids = [doc_id for doc_id, _ in results]
        doc_details = search_system.get_document_details(doc_ids)
        doc_map = {doc['doc_id']: doc for doc in doc_details}
        
        search_results = []
        for doc_id, score in results:
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                search_results.append(SearchResult(
                    doc_id=doc_id,
                    title=doc['title'],
                    author=doc['author'],
                    year=doc['year'],
                    keywords=doc['keywords'],
                    chunk_text=doc['chunk_text'],
                    score=score
                ))
        
        return SearchResponse(
            query=query,
            method="vector_only",
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/keyword_search", response_model=SearchResponse)
async def keyword_search(query: str, k: int = 3):
    """
    Perform keyword search only using SQLite FTS5.
    
    - **query**: Search query string
    - **k**: Number of top results to return (default: 3)
    """
    try:
        results = search_system.keyword_search(query, k)
        
        if not results:
            return SearchResponse(
                query=query,
                method="keyword_only",
                results=[],
                total_results=0
            )
        
        doc_ids = [doc_id for doc_id, _ in results]
        doc_details = search_system.get_document_details(doc_ids)
        doc_map = {doc['doc_id']: doc for doc in doc_details}
        
        search_results = []
        for doc_id, score in results:
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                search_results.append(SearchResult(
                    doc_id=doc_id,
                    title=doc['title'],
                    author=doc['author'],
                    year=doc['year'],
                    keywords=doc['keywords'],
                    chunk_text=doc['chunk_text'],
                    score=score
                ))
        
        return SearchResponse(
            query=query,
            method="keyword_only",
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_documents")
async def add_documents(documents: List[Document]):
    """
    Add new documents to the search index.
    
    - **documents**: List of documents to add
    """
    try:
        doc_dicts = [doc.dict() for doc in documents]
        search_system.add_documents(doc_dicts)
        search_system.save_index()
        
        return {
            "message": f"Successfully added {len(documents)} documents",
            "count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        import sqlite3
        conn = sqlite3.connect(search_system.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        embedding_count = cursor.fetchone()[0]
        
        conn.close()
        
        faiss_count = search_system.faiss_index.ntotal if search_system.faiss_index else 0
        
        return {
            "total_documents": doc_count,
            "total_embeddings": embedding_count,
            "faiss_index_size": faiss_count,
            "model_name": search_system.embedding_manager.model_name,
            "embedding_dimension": search_system.dimension
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Hybrid Search API server...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Alternative docs at: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)