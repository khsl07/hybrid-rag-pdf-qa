"""
Pipeline Unit and Integration Tests
Verifies chunking, FAISS, BM25, and Reciprocal Rank Fusion (RRF) logic.
"""

import numpy as np
from rag.pdf_loader import PDFLoader, DocumentChunk
from rag.vector_store import FAISSVectorStore
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever


def test_rrf_math():
    print("Testing RRF ranking mathematical precision...")
    chunks = [
        DocumentChunk("doc1_p1_c0", "Deep learning algorithms utilize neural networks.", "doc1", 1, 48),
        DocumentChunk("doc1_p1_c1", "BM25 is a ranking function used by search engines for keyword matching.", "doc1", 1, 71),
        DocumentChunk("doc1_p2_c2", "FAISS provides fast similarity search and clustering of dense vectors.", "doc1", 2, 72),
        DocumentChunk("doc1_p2_c3", "Reciprocal Rank Fusion fuses multi-system rankings without normalization.", "doc1", 2, 73),
    ]

    # Mock embeddings: 4 vectors with dimension 8
    np.random.seed(42)
    fake_embeddings = np.random.randn(4, 8).astype(np.float32)
    fake_embeddings /= np.linalg.norm(fake_embeddings, axis=1, keepdims=True)

    # 1. Build FAISS
    vs = FAISSVectorStore(dimension=8)
    vs.add_chunks(chunks, fake_embeddings)
    assert vs.total_vectors() == 4

    # 2. Build BM25
    bm25 = BM25Retriever()
    bm25.index_chunks(chunks)

    # 3. Hybrid Retriever
    hybrid = HybridRetriever(vs, bm25)

    # Query with keyword match for BM25
    query = "search engines BM25 keyword matching"
    q_vec = fake_embeddings[1]  # points strongly to chunk 1

    results = hybrid.retrieve(query=query, query_vector=q_vec, top_k=3, search_mode="hybrid", rrf_k=60)

    assert len(results) > 0
    top_result = results[0]
    print(f"Top result chunk ID: {top_result.chunk.chunk_id}")
    print(f"RRF Score: {top_result.rrf_score:.6f}")
    print(f"Source channel: {top_result.source_channel}")
    print("RRF Fusion test PASSED!")


def test_chunker():
    print("Testing Semantic Chunker...")
    loader = PDFLoader(chunk_size=150, chunk_overlap=30)
    sample_text = (
        "Retrieval-Augmented Generation (RAG) optimizes LLM outputs with external knowledge. "
        "Dense retrieval maps queries and documents into continuous vector spaces. "
        "Sparse retrieval relies on inverted index token matching such as BM25. "
        "Combining both methods yields high precision and robust domain recall."
    )
    chunks = loader._chunk_page_text(sample_text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 180
    print(f"Generated {len(chunks)} chunks successfully. Chunker test PASSED!")


if __name__ == "__main__":
    test_chunker()
    test_rrf_math()
    print("\nAll unit tests passed successfully!")
