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
    ) -> None:
        self._embed = embed_model
        self._store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[str]:
        """
        Retrieve top-k text chunks most relevant to the query.

        Args:
            query: Natural language question or search string.

        Returns:
            Ordered list of chunk strings (most relevant first).
        """
        logger.debug("Retrieving top-%d chunks for query: '%s'", self.top_k, query)

        query_vec = self._embed.encode_query(query)
        results = self._store.search(query_vec, top_k=self.top_k)

        chunks = [r["text"] for r in results]

        if not chunks:
            logger.warning("No relevant chunks found for query.")
        else:
            logger.debug("Retrieved %d chunks (top score: %.4f).", len(results), results[0]["score"])

        return chunks