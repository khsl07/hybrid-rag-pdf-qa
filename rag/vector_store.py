import os
import pickle
import numpy as np
from typing import List, Tuple, Optional
import faiss
from .pdf_loader import DocumentChunk


class FAISSVectorStore:
    """
    Manages FAISS dense vector indexing and cosine similarity search
    coupled with document chunk metadata storage.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        # IndexFlatIP calculates inner products, matching cosine similarity for normalized vectors
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[DocumentChunk] = []

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """
        Adds text embeddings and their associated chunk metadata to the FAISS index.
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings for {len(chunks)} chunks."
            )

        if embeddings.shape[1] != self.dimension:
            # Reinitialize index to match embedding dimensionality if needed
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Performs inner-product (cosine) search against the FAISS index.
        Returns a list of tuples: (DocumentChunk, score, rank_1_indexed).
        """
        if self.index.ntotal == 0:
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results: List[Tuple[DocumentChunk, float, int]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score), rank))

        return results

    def save(self, directory: str):
        """Saves the FAISS index and chunk metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "index.faiss")
        metadata_path = os.path.join(directory, "chunks.pkl")

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "dimension": self.dimension}, f)

    @classmethod
    def load(cls, directory: str) -> "FAISSVectorStore":
        """Loads a saved FAISS index and chunk metadata from disk."""
        index_path = os.path.join(directory, "index.faiss")
        metadata_path = os.path.join(directory, "chunks.pkl")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"FAISS index files not found in {directory}")

        with open(metadata_path, "rb") as f:
            data = pickle.load(f)

        store = cls(dimension=data["dimension"])
        store.index = faiss.read_index(index_path)
        store.chunks = data["chunks"]
        return store

    def total_vectors(self) -> int:
        return self.index.ntotal
