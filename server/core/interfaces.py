from typing import Protocol

from langchain.schema import Document


class BaseRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs,
    ) -> list[Document]:
        """
        Retrieve documents with advanced parameters.

        Args:
            query: Search query
            k: Maximum number of documents to return
            **kwargs: Additional retriever-specific parameters
        """
        ...
