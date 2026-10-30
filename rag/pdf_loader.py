import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional
import pypdf


@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    doc_name: str
    page_number: int
    char_count: int

    def to_dict(self):
        return asdict(self)


class PDFLoader:
    """
    Loads PDF files, extracts text page-by-page, and applies semantic-aware
    sliding window chunking while preserving page-level source attribution.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        """
        Args:
            chunk_size: Target character size for each chunk.
            chunk_overlap: Overlapping characters between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_chunk(self, file_path_or_buffer, doc_name: str) -> List[DocumentChunk]:
        """
        Reads a PDF file or file-like buffer, extracts text per page,
        and returns a list of DocumentChunks with page attribution.
        """
        reader = pypdf.PdfReader(file_path_or_buffer)
        all_chunks: List[DocumentChunk] = []

        chunk_counter = 0
        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1
            raw_text = page.extract_text() or ""
            cleaned_text = self._clean_text(raw_text)

            if not cleaned_text.strip():
                continue

            page_chunks = self._chunk_page_text(cleaned_text)
            for chunk_text in page_chunks:
                chunk_id = f"{doc_name}_p{page_number}_c{chunk_counter}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        doc_name=doc_name,
                        page_number=page_number,
                        char_count=len(chunk_text),
                    )
                )
                chunk_counter += 1

        return all_chunks

    def _clean_text(self, text: str) -> str:
        """Removes extraneous whitespace, control characters, and line breaks."""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        # Fix hyphenated line breaks (e.g., 'arti-\ncle' -> 'article')
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_page_text(self, text: str) -> List[str]:
        """
        Semantic chunker that respects sentence boundaries and paragraph breaks.
        """
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            # If paragraph itself is larger than chunk_size, split by sentences
            if len(para) > self.chunk_size:
                sentences = re.split(r"(?<=[.?!])\s+", para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                        current_chunk = (
                            f"{current_chunk} {sentence}".strip()
                            if current_chunk
                            else sentence
                        )
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            # Apply overlap from the tail of current chunk
                            overlap_text = current_chunk[-self.chunk_overlap :]
                            current_chunk = f"{overlap_text} {sentence}".strip()
                        else:
                            chunks.append(sentence)
                            current_chunk = ""
            else:
                if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                    current_chunk = (
                        f"{current_chunk}\n\n{para}".strip()
                        if current_chunk
                        else para
                    )
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        overlap_text = current_chunk[-self.chunk_overlap :]
                        current_chunk = f"{overlap_text}\n\n{para}".strip()
                    else:
                        chunks.append(para)
                        current_chunk = ""

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks
