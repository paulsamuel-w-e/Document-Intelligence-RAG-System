"""
Document loader: extracts text from PDF using PyMuPDF.
Falls back to PaddleOCR when extracted text is sparse or noisy.
"""

from pathlib import Path

import fitz  # PyMuPDF

from utils.logger import get_logger

logger = get_logger(__name__)

# Heuristics for OCR fallback
_MIN_CHAR_COUNT = 100          # fewer chars than this → likely scanned
_MIN_ALPHA_RATIO = 0.60        # less than 60 % alphabetic → noisy/garbled
_OCR_TRIGGER_PAGE_RATIO = 0.5  # if >50 % of pages are low-quality → full OCR


def _is_low_quality(text: str) -> bool:
    """Return True if the extracted text looks sparse or noisy."""
    if len(text) < _MIN_CHAR_COUNT:
        return True
    alpha_chars = sum(c.isalpha() for c in text)
    ratio = alpha_chars / max(len(text), 1)
    return ratio < _MIN_ALPHA_RATIO


def _extract_with_pymupdf(pdf_path: str) -> tuple[str, int, int]:
    """
    Extract text from all pages using PyMuPDF.

    Returns:
        (full_text, total_pages, low_quality_page_count)
    """
    doc = fitz.open(pdf_path)
    pages_text: list[str] = []
    low_quality_count = 0

    for page in doc:
        text = page.get_text("text")
        if _is_low_quality(text):
            low_quality_count += 1
        pages_text.append(text)

    doc.close()
    return "\n\n".join(pages_text), len(pages_text), low_quality_count


def load_document(file_path: str) -> str:
    """
    Load a PDF document and return its full text.

    Strategy:
      1. Extract text with PyMuPDF.
      2. If more than half the pages are low-quality, fall back to OCR.

    Args:
        file_path: Absolute or relative path to the PDF.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError:        If the file is not a PDF.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported, got: {path.suffix}")

    logger.info("Loading document: %s", path.name)

    text, total_pages, low_quality_pages = _extract_with_pymupdf(file_path)
    low_quality_ratio = low_quality_pages / max(total_pages, 1)

    logger.debug(
        "PyMuPDF: %d pages, %d low-quality (%.0f%%)",
        total_pages,
        low_quality_pages,
        low_quality_ratio * 100,
    )

    if low_quality_ratio > _OCR_TRIGGER_PAGE_RATIO:
        logger.warning(
            "Text quality too low (%.0f%% bad pages). Switching to OCR...",
            low_quality_ratio * 100,
        )
        from ingestion.ocr import extract_text_from_pdf_via_ocr  # lazy import
        text = extract_text_from_pdf_via_ocr(file_path)
        logger.info("OCR complete. Extracted %d characters.", len(text))
    else:
        logger.info(
            "PyMuPDF extraction complete. Extracted %d characters.", len(text)
        )

    return text