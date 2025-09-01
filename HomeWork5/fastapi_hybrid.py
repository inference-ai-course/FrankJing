#!/usr/bin/env python3
"""
FastAPI endpoint for Hybrid Retrieval System
============================================

This module provides a REST API for the hybrid search system,
allowing users to perform vector, keyword, and hybrid searches
via HTTP endpoints.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from hybrid_search import HybridRetrievalSystem, SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Retrieval API",
    description="API for hybrid search combining FAISS vector search and SQLite FTS5 keyword search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global hybrid system instance
hybrid_system = None

class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str
    k: int = 3
    alpha: float = 0.6
    fusion_method: str = "weighted"

class SearchResponse(BaseModel):
    """Response model for search results."""
    query: str
    method: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: float

class EvaluationRequest(BaseModel):
    """Request model for evaluation operations."""
    test_queries: List[Dict[str, Any]]
    k: int = 3

class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    metrics: Dict[str, float]
    total_queries: int

@app.on_event("startup")
async def startup_event():
    """Initialize the hybrid system on startup."""
    global hybrid_system
    try:
        hybrid_system = HybridRetrievalSystem(db_path="hybrid_index.db")
        
        # Try to load existing index
        if not hybrid_system.load_existing_index():
            logger.warning("Failed to load existing index. Please build the index first.")
        else:
            logger.info("Hybrid system initialized successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize hybrid system: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Hybrid Retrieval API",
        "version": "1.0.0",
        "endpoints": {
            "vector_search": "/vector_search",
            "keyword_search": "/keyword_search", 
            "hybrid_search": "/hybrid_search",
            "evaluate": "/evaluate",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Hybrid system not initialized")
    
    return {
        "status": "healthy",
        "system_loaded": hybrid_system.model is not None and hybrid_system.faiss_index is not None,
        "database_path": hybrid_system.db_path
    }

@app.post("/vector_search", response_model=SearchResponse)
async def vector_search(request: SearchRequest):
    """Perform semantic search using FAISS vectors."""
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Hybrid system not initialized")
    
    try:
        import time
        start_time = time.time()
        
        results = hybrid_system.vector_search(request.query, request.k)
        
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Convert results to response format
        result_dicts = []
        for result in results:
            result_dicts.append({
                "chunk_id": result.chunk_id,
                "text": result.text,
                "source_name": result.source_name,
                "source_path": result.source_path,
                "start_token": result.start_token,
                "end_token": result.end_token,
                "vector_score": result.vector_score,
                "rank": result.rank
            })
        
        return SearchResponse(
            query=request.query,
            method="vector",
            results=result_dicts,
            total_results=len(result_dicts),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/keyword_search", response_model=SearchResponse)
async def keyword_search(request: SearchRequest):
    """Perform keyword search using SQLite FTS5."""
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Hybrid system not initialized")
    
    try:
        import time
        start_time = time.time()
        
        results = hybrid_system.keyword_search(request.query, request.k)
        
        search_time = (time.time() - start_time) * 1000
        
        # Convert results to response format
        result_dicts = []
        for result in results:
            result_dicts.append({
                "chunk_id": result.chunk_id,
                "text": result.text,
                "source_name": result.source_name,
                "source_path": result.source_path,
                "start_token": result.start_token,
                "end_token": result.end_token,
                "keyword_score": result.keyword_score,
                "rank": result.rank
            })
        
        return SearchResponse(
            query=request.query,
            method="keyword",
            results=result_dicts,
            total_results=len(result_dicts),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hybrid_search", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """Perform hybrid search combining vector and keyword results."""
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Hybrid system not initialized")
    
    try:
        import time
        start_time = time.time()
        
        results = hybrid_system.hybrid_search(
            request.query, 
            request.k, 
            request.alpha, 
            request.fusion_method
        )
        
        search_time = (time.time() - start_time) * 1000
        
        # Convert results to response format
        result_dicts = []
        for result in results:
            result_dicts.append({
                "chunk_id": result.chunk_id,
                "text": result.text,
                "source_name": result.source_name,
                "source_path": result.source_path,
                "start_token": result.start_token,
                "end_token": result.end_token,
                "vector_score": result.vector_score,
                "keyword_score": result.keyword_score,
                "hybrid_score": result.hybrid_score,
                "rank": result.rank
            })
        
        return SearchResponse(
            query=request.query,
            method=f"hybrid_{request.fusion_method}",
            results=result_dicts,
            total_results=len(result_dicts),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hybrid_search")
async def hybrid_search_get(
    query: str = Query(..., description="Search query"),
    k: int = Query(3, description="Number of results to return"),
    alpha: float = Query(0.6, description="Weight for vector search (0-1)"),
    fusion_method: str = Query("weighted", description="Fusion method: 'weighted' or 'rrf'")
):
    """GET endpoint for hybrid search (convenience endpoint)."""
    request = SearchRequest(
        query=query,
        k=k,
        alpha=alpha,
        fusion_method=fusion_method
    )
    return await hybrid_search(request)

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_search(request: EvaluationRequest):
    """Evaluate search performance on test queries."""
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Hybrid system not initialized")
    
    try:
        metrics = hybrid_system.evaluate_search(request.test_queries, request.k)
        
        return EvaluationResponse(
            metrics=metrics,
            total_queries=len(request.test_queries)
        )
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if hybrid_system is None:
        raise HTTPException(status_code=503, detail="Hybrid system not initialized")
    
    try:
        import sqlite3
        conn = sqlite3.connect(hybrid_system.db_path)
        
        # Get document count
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        
        # Get chunk count
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        
        # Get FAISS index info
        faiss_info = {
            "total_vectors": hybrid_system.faiss_index.ntotal if hybrid_system.faiss_index else 0,
            "dimension": hybrid_system.faiss_index.d if hybrid_system.faiss_index else 0
        }
        
        conn.close()
        
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "faiss_index": faiss_info,
            "model": hybrid_system.metadata.get("model", "unknown") if hybrid_system.metadata else "unknown"
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_hybrid:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
