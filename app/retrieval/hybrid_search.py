class HybridRetriever:
    def __init__(self, vector_store, bm25_retriever, rrf_k=60):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query, query_embedding, top_k=3):
        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        sparse_results = self.bm25_retriever.search(
            query=query,
            top_k=top_k
        )

        scores = {}
        chunks = {}

        for results in [dense_results, sparse_results]:
            for rank, result in enumerate(results, start=1):
                chunk = result["chunk"]
                chunk_id = chunk["chunk_id"]

                scores[chunk_id] = (
                    scores.get(chunk_id, 0)
                    + 1 / (self.rrf_k + rank)
                )

                chunks[chunk_id] = chunk

        ranked_results = sorted(
            [
                {
                    "chunk": chunks[chunk_id],
                    "score": score
                }
                for chunk_id, score in scores.items()
            ],
            key=lambda item: item["score"],
            reverse=True
        )

        return ranked_results[:top_k]
