"""
Retriever: converts a user query to an embedding and fetches top-k chunks.
Thin glue between EmbeddingModel and VectorStore.
"""

from rag.embeddings import EmbeddingModel
from rag.vectorstore import VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Given a query string, retrieve the most relevant document chunks.

    Args:
        embed_model:  An initialised EmbeddingModel.
        vector_store: A populated VectorStore.
        top_k:        Number of chunks to return per query.
    """

    def __init__(
        self,
        embed_model: EmbeddingModel,
        vector_store: VectorStore,
        top_k: int = 5,
        reranker = None
    ) -> None:
        self._embed = embed_model
        self._store = vector_store
        self.top_k = top_k
        self._reranker = reranker

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """
        Retrieve top-k text chunks most relevant to the query.

        Args:
            query: Natural language question or search string.

        Returns:
            Ordered list of chunk strings (most relevant first).
        """
        logger.debug("Retrieving top-%d chunks for query: '%s'", self.top_k, query)

        query_vec = self._embed.encode_query(query)
        k = top_k if top_k is not None else self.top_k
        final_k = k
        results = self._store.search(query_vec, top_k=k * 2)  # extra for filtering/rerank

        # NEW: filter weak matches
        MIN_SCORE = 0.3  # tune later
        filtered = [r for r in results if r["score"] >= MIN_SCORE]

        # fallback if everything removed
        if not filtered:
            filtered = results

        chunks = [r["text"] for r in filtered]

        # Step 3: rerank
        if self._reranker:
            ranked = self._reranker.rerank(query, chunks)
            chunks = [c for c, _ in ranked[:final_k]]

        if not chunks:
            logger.warning("No relevant chunks found for query.")
        else:
            logger.debug("Retrieved %d chunks (top score: %.4f).", len(results), results[0]["score"])

        return chunks