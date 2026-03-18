"""
Retriever: converts a user query to an embedding and fetches top-k chunks.
Thin glue between EmbeddingModel and VectorStore.
"""

from rag.embeddings import EmbeddingModel
from rag.vectorstore import VectorStore
from rag.bm25 import BM25Retriever
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
        self._bm25 = BM25Retriever(vector_store._chunks)

    def _retrieve_internal(self, query: str, top_k: int | None = None) -> list[str]:
        """
        Retrieve top-k text chunks most relevant to the query.

        Args:
            query: Natural language question or search string.

        Returns:
            Ordered list of chunk strings (most relevant first).
        """

        k = top_k if top_k else self.top_k
        logger.debug("Retrieving top-%d chunks for query: '%s'", k, query)
        # --- Dense search ---
        query_vec = self._embed.encode_query(query)
        dense_results = self._store.search(query_vec, top_k=k * 2)

        # --- Sparse search ---
        sparse_results = self._bm25.search(query, top_k=k * 2)

        # --- Merge ---
        combined = {}

        for r in dense_results:
            idx = r["index"]
            combined[idx] = {
                "text": r["text"],
                "section": r.get("section", "body"),
                "score": r["score"]
            }

        for r in sparse_results:
            idx = r["index"]
            if idx in combined:
                combined[idx]["score"] += r["score"]
            else:
                combined[idx] = {
                    "text": r["text"],
                    "section": r.get("section", "body"),
                    "score": r["score"]
                }

        # --- Metadata weighting ---
        for item in combined.values():
            if item["section"] == "related":
                item["score"] *= 0.5

        # --- Convert ---
        merged = list(combined.values())

        # --- Sort ---
        merged = sorted(merged, key=lambda x: x["score"], reverse=True)

        chunks = [r["text"] for r in merged[:k * 2]]

        # --- Rerank ---
        if self._reranker:
            ranked = self._reranker.rerank(query, chunks)
            final_chunks = [c for c, _ in ranked[:k]]
        else:
            final_chunks = chunks[:k]

        if not final_chunks:
            logger.warning("No relevant chunks found for query.")
        else:
            logger.debug("Retrieved %d chunks.", len(final_chunks))

        return final_chunks, merged[:k]
    
    def retrieve(self, query: str, top_k: int | None = None):
        chunks, _ = self._retrieve_internal(query, top_k)
        return chunks
    
    def retrieve_with_metadata(self, query: str, top_k: int | None = None):
        return self._retrieve_internal(query, top_k)