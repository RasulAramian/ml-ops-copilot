# ML Ops Copilot
## Evidence-Grounded RAG Assistant

A RAG application for answering machine learning documentation questions.

### Features
- PDF, TXT, Markdown ingestion
- Sentence-aware chunking
- Sentence Transformers embeddings
- Qdrant dense retrieval
- BM25 sparse retrieval
- Reciprocal Rank Fusion
- Cross-Encoder reranking
- Qwen2.5-1.5B generation
- Evidence-grounded answers
- Source attribution
- Abstention for unsupported questions
- FastAPI and Docker
- Automated pytest tests

### Architecture
Documents -> Chunking -> Embeddings -> Dense and Sparse Retrieval
-> Hybrid Search -> Reranking -> Context -> LLM -> Answer

### Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API documentation: http://localhost:8000/docs

### Tests
```bash
python -m pytest -q
```

Verified result: 3 passed.

### Limitations
Portfolio prototype, not a production deployment.
Qdrant uses in-memory storage. Documents are not persisted.
The sample knowledge base currently contains LightGBM documentation.
Evaluation uses basic rule-based checks.

### Author
Rasul Aramian

GitHub: https://github.com/RasulAramian
LinkedIn: https://www.linkedin.com/in/rasul-aramian-958010115/
