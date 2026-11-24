from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model=None
    ):
        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
