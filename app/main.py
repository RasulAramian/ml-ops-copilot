
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.rag_factory import create_rag_pipeline


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question")


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]


def create_app(pipeline=None):

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None:
            app.state.pipeline = create_rag_pipeline()
        yield

    app = FastAPI(
        title="ML Ops Copilot",
        description="Evidence-grounded RAG API for ML engineering documentation.",
        version="1.0.0",
        lifespan=lifespan
    )

    app.state.pipeline = pipeline

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "pipeline_ready": app.state.pipeline is not None
        }

    @app.post("/query", response_model=QueryResponse)
    def query_documents(request: QueryRequest):
        if app.state.pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline is not ready."
            )

        try:
            result = app.state.pipeline.run(request.query)
            return {
                "query": result["query"],
                "answer": result["answer"],
                "sources": result["sources"]
            }
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=str(error)
            )

    return app


app = create_app()
