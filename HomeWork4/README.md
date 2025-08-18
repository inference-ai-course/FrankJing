Week 4: Retrieval‑Augmented Generation (RAG) with Local PDFs (./data)
=====================================================================
The PDF files are downloaded with downloadpaper.py file.

This single file builds a complete RAG pipeline over the PDF files located
under ./data. It supports:

1) Data Collection: discover PDFs in ./data (recursively).
2) Text Extraction: extract text using PyMuPDF (fitz).
3) Chunking: sliding‑window word chunks (default 512, overlap 50).
4) Embeddings: Sentence-Transformers (default: all-MiniLM-L6-v2, dim=384).
5) Indexing: FAISS IndexFlatL2 with persistence.
6) Query: CLI demo retrieval and a FastAPI service at /search.
7) Retrieval Report: generate a small report for multiple queries.

Quick start
-----------
# Create venv and install deps (Windows users MUST use faiss-cpu)
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install faiss-cpu sentence-transformers PyMuPDF fastapi uvicorn tqdm numpy
# PyTorch (required by sentence-transformers). CPU wheels:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Ensure PDFs are placed under ./data

# Build index (extract -> chunk -> embed -> index)
python rag_local_pdfs.py build --data ./data --chunk-size 512 --overlap 50 --model all-MiniLM-L6-v2

# Try a quick query (CLI)
python rag_local_pdfs.py query --q "What is contrastive learning in NLP?" --k 3

# Serve FastAPI
uvicorn rag_local_pdfs:app --host 0.0.0.0 --port 8000 --reload
# Visit: http://localhost:8000/search?q=contrastive%20learning

# Generate a simple report
python rag_local_pdfs.py report --queries "['What is a transformer?', 'What is RAG?', 'How do adapters work?', 'Contrastive learning objective', 'BLEU vs ROUGE']"

Artifacts
---------
.artifacts/
  texts/                 # extracted plain text per PDF
  chunks.jsonl           # one JSON per chunk
  embeddings.npy         # float32 [num_chunks, dim]
  meta.json              # metadata (model, dim, time, params)
  faiss.index            # FAISS binary index