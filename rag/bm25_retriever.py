import os
import re
import pickle
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from .pdf_loader import DocumentChunk


class BM25Retriever:
    """
    Sparse keyword-based retrieval engine implementing BM25Okapi.
    Complements dense embedding retrieval by identifying exact matches,
    proper nouns, technical IDs, acronyms, and specialized terminology.
    """

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunks: List[DocumentChunk] = []
        self.corpus_size: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple whitespace & alphanumeric tokenizer with lowercasing."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Builds the BM25 inverted index from document chunks."""
        self.chunks = chunks
        self.corpus_size = len(chunks)
        tokenized_corpus = [self.tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Executes BM25 scoring over indexed chunks.
        Returns a list of tuples: (DocumentChunk, score, rank_1_indexed).
        """
        if not self.bm25 or self.corpus_size == 0:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get indices sorted by score descending
        ranked_indices = [
            i for i in sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
            if doc_scores[i] > 0
        ][:top_k]

        results: List[Tuple[DocumentChunk, float, int]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append((self.chunks[idx], float(doc_scores[idx]), rank))

        return results

    def save(self, directory: str):
        """Saves BM25 index and chunk references to disk."""
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "bm25.pkl")
        with open(file_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, directory: str) -> "BM25Retriever":
        """Loads BM25 index from disk."""
        file_path = os.path.join(directory, "bm25.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"BM25 file not found at {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        retriever = cls()
        retriever.chunks = data["chunks"]
        retriever.bm25 = data["bm25"]
        retriever.corpus_size = len(retriever.chunks)
        return retriever
