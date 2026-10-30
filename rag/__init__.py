"""
Hybrid RAG PDF Q&A System
Combining FAISS Dense Semantic Search + BM25 Lexical Keyword Search via Reciprocal Rank Fusion (RRF)
with Grounded Gemini API Generation.
"""

from .pdf_loader import PDFLoader, DocumentChunk
from .embeddings import GeminiEmbedder
from .vector_store import FAISSVectorStore
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever, RetrievalResult
from .generator import GroundedGenerator

__all__ = [
    "PDFLoader",
    "DocumentChunk",
    "GeminiEmbedder",
    "FAISSVectorStore",
    "BM25Retriever",
    "HybridRetriever",
    "RetrievalResult",
    "GroundedGenerator",
]
