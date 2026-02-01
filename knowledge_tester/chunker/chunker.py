from abc import ABC, abstractmethod

from langchain.schema.document import Document


class Chunker(ABC):
    """
    Description: An abstract base class for chunking documents into smaller parts. This base class provide a strict interface for all chunker implementations.
    """

    @classmethod
    @abstractmethod
    def chunk(cls, documents: list[Document]) -> list[Document]:
        """
        Description: Abstract method to chunk a list of documents into smaller parts.
        Args:
            documents (list[Document]): A list of documents to be chunked.

        Returns:
            list[Document]: A list of chunked documents.
        """
        pass
