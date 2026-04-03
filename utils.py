# utils.py
import re
from typing import List

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace and remove control characters
    text = re.sub(r"\s+", " ", text).strip()
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t ")
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Word-based chunking with overlap. Good default for MPNet-style encoders.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks





