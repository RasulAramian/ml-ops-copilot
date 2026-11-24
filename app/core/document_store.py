
from uuid import uuid4


class DocumentStore:
    def __init__(self):
        self.documents = {}

    def create(self, document_data):
        document_id = str(uuid4())

        document = {
            "id": document_id,
            **document_data
        }

        self.documents[document_id] = document
        return document

    def get(self, document_id):
        return self.documents.get(document_id)
