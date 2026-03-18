"""
OCR module using PaddleOCR.
Lazy-loaded to avoid import overhead when OCR is not needed.
"""

from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton — only instantiated on first use
_ocr_engine = None


def _get_engine():
    """Lazy-initialize PaddleOCR exactly once per process."""
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Initializing PaddleOCR engine (first use)...")
        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            logger.info("PaddleOCR engine ready.")
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. Run: pip install paddleocr"
            ) from exc
    return _ocr_engine


def extract_text_from_image(image_path: str) -> str:
    """
    Run OCR on a single image file.

    Args:
        image_path: Path to the image (PNG, JPEG, etc.)

    Returns:
        Extracted text as a single string.
    """
    engine = _get_engine()
    result = engine.ocr(image_path, cls=True)

    if not result or not result[0]:
        logger.warning("OCR returned no results for %s", image_path)
        return ""

    lines = [word_info[1][0] for line in result for word_info in line]
    text = "\n".join(lines)
    logger.debug("OCR extracted %d characters from %s", len(text), image_path)
    return text


def extract_text_from_pdf_via_ocr(pdf_path: str, dpi: int = 150) -> str:
    """
    Convert each page of a PDF to an image, then run OCR.
    Used as a fallback when PyMuPDF yields low-quality text.

    Args:
        pdf_path: Path to the PDF file.
        dpi:      Rendering resolution (higher = better quality, slower).

    Returns:
        Concatenated OCR text for all pages.
    """
    import tempfile
    import fitz  # PyMuPDF  # noqa: PLC0415

    doc = fitz.open(pdf_path)
    all_text: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for page_num, page in enumerate(doc):
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_path = str(Path(tmp_dir) / f"page_{page_num:04d}.png")
            pix.save(img_path)

            logger.debug("Running OCR on page %d of %s", page_num + 1, pdf_path)
            page_text = extract_text_from_image(img_path)
            all_text.append(page_text)

    doc.close()
    return "\n\n".join(all_text)