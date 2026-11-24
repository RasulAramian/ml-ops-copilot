
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model=None
    ):
        self.model = model or CrossEncoder(model_name)

    def rerank(self, query, results, top_k=3):
        if not results:
            return []

        pairs = [
            (query, result["chunk"]["text"])
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            reranked_results.append({
                "chunk": result["chunk"],
                "score": float(score)
            })

        reranked_results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return reranked_results[:top_k]
