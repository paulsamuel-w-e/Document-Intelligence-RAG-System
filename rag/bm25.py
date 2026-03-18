from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(range(len(self.chunks)), self.chunks, scores),
            key=lambda x: x[2],
            reverse=True
        )

        return [
            {"text": c["text"], "score": float(s), "index": idx}
            for idx, c, s in ranked[:top_k]
        ]