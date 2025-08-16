import time

from core.interfaces import BaseRetriever
from langchain.schema import Document


class MockBaseRetriever(BaseRetriever):
    """
    Mock database retriever for testing and development.
    Simulates vector search with predefined responses.
    """

    def __init__(self, mock_documents: list[Document] | None = None, delay: float = 0.1):
        """
        Initialize mock retriever with optional predefined documents.

        Args:
            mock_documents: Predefined documents to return
            delay: Simulated retrieval delay in seconds
        """
        self.delay = delay
        self.mock_documents = mock_documents or self._create_default_documents()

    def _create_default_documents(self) -> list[Document]:
        """Create some default mock documents for testing."""
        return [
            Document(
                page_content="This is a sample document about artificial intelligence and machine learning.",
                metadata={"source": "mock_doc_1.pdf", "page": 1, "type": "internal"},
            ),
            Document(
                page_content="Python programming language is widely used for data science and AI development.",
                metadata={"source": "mock_doc_2.pdf", "page": 1, "type": "internal"},
            ),
            Document(
                page_content="LangChain is a framework for developing applications powered by language models.",
                metadata={"source": "mock_doc_3.pdf", "page": 1, "type": "internal"},
            ),
            Document(
                page_content="Vector databases enable efficient similarity search for large document collections.",
                metadata={"source": "mock_doc_4.pdf", "page": 1, "type": "internal"},
            ),
            Document(
                page_content="Natural language processing involves understanding and generating human language.",
                metadata={"source": "mock_doc_5.pdf", "page": 1, "type": "internal"},
            ),
        ]

    def retrieve(self, query: str, k: int = 5, **kwargs) -> list[Document]:
        """
        Mock document retrieval with simple keyword matching.

        Args:
            query: Search query
            k: Maximum number of documents to return
            **kwargs: Additional parameters (ignored in mock)

        Returns:
            List of mock documents
        """
        # Simulate retrieval delay
        if self.delay > 0:
            time.sleep(self.delay)

        query_words = set(query.lower().split())
        scored_docs = []

        for doc in self.mock_documents:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words))

            score = overlap / max(len(query_words), 1)
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]
