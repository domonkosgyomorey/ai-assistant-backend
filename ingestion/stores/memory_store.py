from langchain.schema import Document

from ingestion.interfaces import DocumentStore


class InMemoryStore(DocumentStore):
    def __init__(self):
        self.docs = []

    def save(self, docs: list[Document]):
        self.docs.extend(docs)

    def list_source_keys(self) -> list[str]:
        return list(set({doc.metadata.get("file_path") for doc in self.docs if doc.metadata.get("file_path")}))
