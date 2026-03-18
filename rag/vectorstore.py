"""
FAISS-backed vector store.
Stores embeddings alongside their source text chunks.
Supports incremental addition and top-k similarity search.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    In-memory FAISS index with an aligned list of text chunks.

    The index uses inner-product (IP) similarity, which is equivalent
    to cosine similarity when embeddings are L2-normalized.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[str] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, chunks: list[str], embeddings: np.ndarray) -> None:
        """
        Add text chunks and their pre-computed embeddings to the store.

        Args:
            chunks:     List of raw text strings.
            embeddings: Float32 array of shape (len(chunks), dimension).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info(
            "Added %d chunks. Total in store: %d", len(chunks), len(self._chunks)
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Return the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: Float32 array of shape (1, dimension).
            top_k:           Number of results to return.

        Returns:
            List of dicts with keys: ``text``, ``score``, ``index``.
        """
        if self._index.ntotal == 0:
            logger.warning("Vector store is empty. No results returned.")
            return []

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {"text": self._chunks[idx], "score": float(score), "index": int(idx)}
            )

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Persist the FAISS index and chunk list to disk."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(dir_path / "index.faiss"))
        with open(dir_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

        logger.info("Vector store saved to %s", directory)

    @classmethod
    def load(cls, directory: str, dimension: int) -> "VectorStore":
        """Restore a previously saved VectorStore from disk."""
        dir_path = Path(directory)
        instance = cls(dimension)
        instance._index = faiss.read_index(str(dir_path / "index.faiss"))
        with open(dir_path / "chunks.pkl", "rb") as f:
            instance._chunks = pickle.load(f)

        logger.info(
            "Vector store loaded from %s (%d chunks).", directory, len(instance._chunks)
        )
        return instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._index.ntotal