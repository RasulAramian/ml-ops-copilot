
from app.generation.context_builder import build_context


class RAGPipeline:
    def __init__(
        self,
        embedding_model,
        hybrid_retriever,
        reranker,
        llm_client,
        retrieval_top_k=5,
        context_top_k=3
    ):
        self.embedding_model = embedding_model
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.retrieval_top_k = retrieval_top_k
        self.context_top_k = context_top_k

    def run(self, query):

        # Step 1: Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Step 2: Retrieve relevant chunks using hybrid search
        retrieved_results = self.hybrid_retriever.search(
            query=query,
            query_embedding=query_embedding,
            top_k=self.retrieval_top_k
        )

        # Step 3: Rerank retrieved chunks
        reranked_results = self.reranker.rerank(
            query=query,
            results=retrieved_results,
            top_k=self.context_top_k
        )

        # Step 4: Build context
        context = build_context(
            reranked_results,
            top_k=self.context_top_k
        )

        # Step 5: Generate grounded answer
        answer = self.llm_client.generate(
            query=query,
            context=context
        )

        # Step 6: Collect source information
        sources = [
            {
                "filename": result["chunk"]["filename"],
                "chunk_id": result["chunk"]["chunk_id"],
                "score": result["score"]
            }
            for result in reranked_results
        ]

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "sources": sources
        }
