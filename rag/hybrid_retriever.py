from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
from .pdf_loader import DocumentChunk
from .vector_store import FAISSVectorStore
from .bm25_retriever import BM25Retriever


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    rrf_score: float
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    source_channel: str = "hybrid"  # "hybrid", "dense_only", or "sparse_only"


class HybridRetriever:
    """
    Fuses dense semantic search (FAISS) and sparse lexical search (BM25)
    using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = w_dense * (1 / (k + rank_dense(d))) + w_sparse * (1 / (k + rank_sparse(d)))

    Where:
        k: Smoothing constant (typically 60) preventing high-ranked outliers from dominating.
        w_dense, w_sparse: Weights controlled by parameter alpha in [0.0, 1.0].
    """

    def __init__(self, vector_store: FAISSVectorStore, bm25_retriever: BM25Retriever):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def retrieve(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        search_mode: str = "hybrid",
        rrf_k: int = 60,
        alpha: float = 0.5,
        candidate_multiplier: int = 3,
    ) -> List[RetrievalResult]:
        """
        Executes retrieval based on the chosen mode: 'hybrid', 'dense', or 'sparse'.

        Args:
            query: Raw text query string for BM25.
            query_vector: Gemini embedding vector for FAISS.
            top_k: Number of final fused results to return.
            search_mode: 'hybrid', 'dense', or 'sparse'.
            rrf_k: RRF smoothing constant (default: 60).
            alpha: Weight for dense vs sparse (1.0 = pure dense, 0.0 = pure sparse, 0.5 = balanced).
            candidate_multiplier: Fetch top_k * candidate_multiplier from each retriever before fusion.
        """
        n_candidates = max(top_k * candidate_multiplier, 10)

        if search_mode == "dense":
            return self._dense_search(query_vector, top_k)
        elif search_mode == "sparse":
            return self._sparse_search(query, top_k)
        else:
            return self._rrf_fusion(query, query_vector, top_k, rrf_k, alpha, n_candidates)

    def _dense_search(self, query_vector: np.ndarray, top_k: int) -> List[RetrievalResult]:
        dense_results = self.vector_store.search(query_vector, top_k=top_k)
        return [
            RetrievalResult(
                chunk=chunk,
                rrf_score=score,
                dense_rank=rank,
                dense_score=score,
                source_channel="dense_only",
            )
            for chunk, score, rank in dense_results
        ]

    def _sparse_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        sparse_results = self.bm25_retriever.search(query, top_k=top_k)
        return [
            RetrievalResult(
                chunk=chunk,
                rrf_score=score,
                sparse_rank=rank,
                sparse_score=score,
                source_channel="sparse_only",
            )
            for chunk, score, rank in sparse_results
        ]

    def _rrf_fusion(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int,
        rrf_k: int,
        alpha: float,
        n_candidates: int,
    ) -> List[RetrievalResult]:
        # 1. Fetch dense candidates
        dense_candidates = self.vector_store.search(query_vector, top_k=n_candidates)

        # 2. Fetch sparse candidates
        sparse_candidates = self.bm25_retriever.search(query, top_k=n_candidates)

        # Map candidate chunks by chunk_id
        chunk_map: Dict[str, DocumentChunk] = {}
        dense_meta: Dict[str, tuple] = {}   # chunk_id -> (score, rank)
        sparse_meta: Dict[str, tuple] = {}  # chunk_id -> (score, rank)
        rrf_scores: Dict[str, float] = {}

        w_dense = alpha * 2.0
        w_sparse = (1.0 - alpha) * 2.0

        # Process dense candidates
        for chunk, score, rank in dense_candidates:
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_meta[cid] = (score, rank)
            dense_rrf = w_dense * (1.0 / (rrf_k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_rrf

        # Process sparse candidates
        for chunk, score, rank in sparse_candidates:
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            sparse_meta[cid] = (score, rank)
            sparse_rrf = w_sparse * (1.0 / (rrf_k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + sparse_rrf

        # Sort all chunks by fused RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_cids = sorted_cids[:top_k]

        fused_results: List[RetrievalResult] = []
        for cid in top_cids:
            has_dense = cid in dense_meta
            has_sparse = cid in sparse_meta

            channel = "both" if (has_dense and has_sparse) else ("dense_only" if has_dense else "sparse_only")

            fused_results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    rrf_score=rrf_scores[cid],
                    dense_rank=dense_meta[cid][1] if has_dense else None,
                    dense_score=dense_meta[cid][0] if has_dense else None,
                    sparse_rank=sparse_meta[cid][1] if has_sparse else None,
                    sparse_score=sparse_meta[cid][0] if has_sparse else None,
                    source_channel=channel,
                )
            )

        return fused_results
