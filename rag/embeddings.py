import os
import numpy as np
from typing import List, Optional


class GeminiEmbedder:
    """
    Handles text embeddings generation via the Google Gemini API (text-embedding-004).
    Produces normalized L2 vectors suitable for cosine similarity in FAISS.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/text-embedding-004",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._client = None
        self._legacy_genai = None

        if self.api_key:
            self._init_client()

    def _init_client(self):
        """Initializes the Gemini API client using available SDKs."""
        try:
            # Try newer google-genai SDK first
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except ImportError:
            try:
                # Fallback to google-generativeai SDK
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=self.api_key)
                self._legacy_genai = genai_legacy
            except ImportError:
                raise ImportError(
                    "Neither 'google-genai' nor 'google-generativeai' is installed. "
                    "Run 'pip install google-genai' or 'pip install google-generativeai'."
                )

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds a list of texts in batches and returns a 2D float32 numpy array.
        Each vector is L2-normalized so dot products equal cosine similarities.
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        if not self._client and not self._legacy_genai:
            self._init_client()

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        arr = np.array(all_embeddings, dtype=np.float32)
        # L2-normalize vectors for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = arr / norms
        return normalized

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single user query string into a normalized 1D float32 vector."""
        vecs = self.embed_texts([query])
        return vecs[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Invokes Gemini API for a single batch."""
        if self._client:
            # New google.genai SDK
            response = self._client.models.embed_content(
                model=self.model_name,
                contents=texts,
            )
            # Response may contain .embeddings list
            return [e.values for e in response.embeddings]
        elif self._legacy_genai:
            # Legacy google.generativeai SDK
            response = self._legacy_genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type="retrieval_document",
            )
            embeddings = response.get("embedding", [])
            # If batch, embedding is a list of lists; if single, list of floats
            if embeddings and isinstance(embeddings[0], (int, float)):
                return [embeddings]
            return embeddings
        else:
            raise RuntimeError("Gemini client is not initialized.")
