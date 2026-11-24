
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.ingestion.document_loader import load_document
from app.ingestion.text_chunker import sentence_aware_chunking

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker

from app.generation.llm_client import LLMClient
from app.generation.rag_pipeline import RAGPipeline


def create_rag_pipeline(model=None, tokenizer=None):

    project_root = Path(__file__).resolve().parents[2]

    # 1. Load sample documents
    sample_path = project_root / "data" / "sample" / "lightgbm.md"

    document = load_document(sample_path)

    chunks = sentence_aware_chunking(
        text=document["content"],
        document_id="sample-lightgbm",
        filename=document["filename"],
        chunk_size=500
    )

    if not chunks:
        raise ValueError("No chunks were created from the sample document.")

    # 2. Initialize embedding model
    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks]
    )

    # 3. Initialize vector store
    vector_store = VectorStore(
        vector_size=embeddings.shape[1]
    )

    vector_store.add_chunks(chunks, embeddings)

    # 4. Initialize sparse retrieval
    bm25_retriever = BM25Retriever(chunks)

    # 5. Initialize hybrid retrieval
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever
    )

    # 6. Initialize reranker
    reranker = CrossEncoderReranker()

    # 7. Load language model if not supplied
    if model is None or tokenizer is None:

        model_name = os.getenv(
            "LLM_MODEL_NAME",
            "Qwen/Qwen2.5-1.5B-Instruct"
        )

        hf_token = os.getenv("HF_TOKEN")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )

        model_dtype = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            dtype=model_dtype,
            device_map="auto"
        )

        model.eval()

    # 8. Initialize LLM client
    llm_client = LLMClient(
        model=model,
        tokenizer=tokenizer
    )

    # 9. Assemble complete RAG pipeline
    return RAGPipeline(
        embedding_model=embedding_model,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm_client=llm_client,
        retrieval_top_k=5,
        context_top_k=3
    )
