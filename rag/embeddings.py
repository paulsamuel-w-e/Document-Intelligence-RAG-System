"""
Embedding wrapper using sentence-transformers.
Encapsulates model loading and batch encoding behind a clean interface.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Thin wrapper around a SentenceTransformer model.

    Handles:
      - Lazy/eager model loading
      - Batch encoding of text lists
      - Single-query encoding
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of texts into dense embedding vectors.

        Args:
            texts:      List of strings to encode.
            batch_size: Number of texts per batch (affects memory).

        Returns:
            Float32 numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine-friendly unit vectors
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.

        Returns:
            Float32 numpy array of shape (1, dimension).
        """
        return self.encode([query])