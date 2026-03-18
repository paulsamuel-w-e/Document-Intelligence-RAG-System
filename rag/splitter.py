"""
Text splitting and cleaning for RAG ingestion.
Uses LangChain's RecursiveCharacterTextSplitter for robust chunking.
Applies heuristic filters to remove low-signal noise.
"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger import get_logger

logger = get_logger(__name__)

# Splitter configuration
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 100

# Filters
_MIN_CHUNK_LENGTH = 80       # discard extremely short chunks
_MIN_ALPHA_RATIO = 0.55      # discard symbol-heavy chunks

# Regex patterns for noise removal
_REFERENCE_PATTERN = re.compile(
    r"references\s*\n.*", re.IGNORECASE | re.DOTALL
)
_FIGURE_CAPTION_PATTERN = re.compile(
    r"(figure|fig\.?|table)\s+\d+[:\.\s][^\n]*\n?", re.IGNORECASE
)
_FOOTER_HEADER_PATTERN = re.compile(
    r"^\s*\d+\s*$", re.MULTILINE  # standalone page numbers
)
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """
    Apply preprocessing to raw extracted text.
    Removes references section, figure captions, page numbers,
    and collapses excessive whitespace.
    """
    text = _REFERENCE_PATTERN.sub("", text)
    text = _FIGURE_CAPTION_PATTERN.sub("", text)
    text = _FOOTER_HEADER_PATTERN.sub("", text)
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)
    return text.strip()


def _is_valid_chunk(chunk: str) -> bool:
    """Return True if a chunk contains enough signal to be useful."""
    if len(chunk.strip()) < _MIN_CHUNK_LENGTH:
        return False
    alpha_count = sum(c.isalpha() for c in chunk)
    ratio = alpha_count / max(len(chunk), 1)
    return ratio >= _MIN_ALPHA_RATIO

def _detect_section(text: str) -> str:
    t = text.lower()

    if "abstract" in t[:200]:
        return "abstract"
    if "introduction" in t:
        return "intro"
    if "related work" in t:
        return "related"
    return "body"


def split_text(text: str) -> list[dict]:
    """
    Clean and split raw document text into overlapping chunks.

    Args:
        text: Raw text extracted from a document.

    Returns:
        List of clean, filtered text chunks ready for embedding.
    """
    cleaned = _clean_text(text)
    logger.debug("Cleaned text length: %d characters", len(cleaned))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(cleaned)
    valid_chunks = [
        {
            "text": c,
            "section": _detect_section(c)
        }
        for c in raw_chunks
        if _is_valid_chunk(c)
    ]

    logger.info(
        "Split into %d chunks (%d discarded as noisy).",
        len(valid_chunks),
        len(raw_chunks) - len(valid_chunks),
    )

    return valid_chunks