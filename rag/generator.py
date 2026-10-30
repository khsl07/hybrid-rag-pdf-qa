import os
from typing import List, Generator, Optional
from .hybrid_retriever import RetrievalResult


SYSTEM_PROMPT = """You are an expert AI document research assistant. Your task is to generate grounded, factually accurate, and source-attributed answers based strictly on the provided context passages retrieved from documents.

Rules:
1. Rely ONLY on facts stated in the CONTEXT. Do not extrapolate, assume, or hallucinate external knowledge.
2. If the context does not contain enough information to address the question, clearly state: "The provided document does not contain sufficient information to answer this question."
3. Every major claim or factual statement MUST be attributed to its source document and page number using citations formatted as [Page X] or [DocName, Page X].
4. Format your response cleanly using GitHub-flavored Markdown (bullet points, bold text for key concepts, tables or code blocks where applicable).
5. At the end of your answer, provide a brief 'Sources Consulted' section listing the documents and pages referenced.
"""


class GroundedGenerator:
    """
    Synthesizes grounded answers using Google's Gemini models
    conditioned strictly on the retrieved document context.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._client = None
        self._legacy_genai = None

        if self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except ImportError:
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=self.api_key)
                self._legacy_genai = genai_legacy
            except ImportError:
                raise ImportError(
                    "Neither 'google-genai' nor 'google-generativeai' is installed."
                )

    def _format_context(self, results: List[RetrievalResult]) -> str:
        """Formats retrieved chunks with clear metadata delimiters for prompting."""
        context_blocks = []
        for i, res in enumerate(results, start=1):
            chunk = res.chunk
            header = f"--- [Passage {i}] Source: {chunk.doc_name} | Page: {chunk.page_number} | Relevance Channel: {res.source_channel} ---"
            block = f"{header}\n{chunk.content}\n"
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        temperature: float = 0.2,
    ) -> str:
        """
        Generates a grounded answer from context passages.
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        if not self._client and not self._legacy_genai:
            self._init_client()

        context_str = self._format_context(retrieval_results)

        prompt = f"""CONTEXT:
{context_str}

USER QUESTION:
{query}

GROUNDED ANSWER (with [Page X] citations):"""

        if self._client:
            # Modern google-genai SDK
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": temperature,
                },
            )
            return response.text
        elif self._legacy_genai:
            # Legacy google-generativeai SDK
            model = self._legacy_genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=SYSTEM_PROMPT,
                generation_config={"temperature": temperature},
            )
            response = model.generate_content(prompt)
            return response.text
        else:
            raise RuntimeError("Gemini client is not initialized.")
