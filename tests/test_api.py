from fastapi.testclient import TestClient

from app.main import create_app


class FakePipeline:
    def run(self, query):
        return {
            "query": query,
            "answer": "GOSS and EFB.",
            "context": "Test context",
            "sources": [
                {
                    "filename": "lightgbm.md",
                    "chunk_id": "test-chunk",
                    "score": 0.9
                }
            ]
        }


def test_health_endpoint():
    app = create_app(pipeline=FakePipeline())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["pipeline_ready"] is True


def test_query_endpoint():
    app = create_app(pipeline=FakePipeline())

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "What are GOSS and EFB?"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "GOSS and EFB."
    assert data["sources"][0]["filename"] == "lightgbm.md"


def test_query_rejects_short_question():
    app = create_app(pipeline=FakePipeline())

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "Hi"}
        )

    assert response.status_code == 422
