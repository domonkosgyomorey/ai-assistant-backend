import random
import time

from core.interfaces import WebSearchRetriever
from langchain.schema import Document


class MockWebSearchRetriever(WebSearchRetriever):
    """
    Mock web search retriever for testing and development.
    Simulates web search with predefined responses.
    """

    def __init__(self, delay: float = 0.2, mock_results: list[Document] | None = None):
        """
        Initialize mock web search retriever.

        Args:
            delay: Simulated search delay in seconds
            mock_results: Predefined search results
        """
        self.delay = delay
        self.mock_results = mock_results or self._create_default_results()

    def _create_default_results(self) -> list[Document]:
        """Create some default mock search results."""
        return [
            Document(
                page_content="Latest developments in artificial intelligence show promising results in natural language processing and computer vision applications.",
                metadata={
                    "source": "https://example.com/ai-news-1",
                    "title": "AI Breakthrough in Natural Language Processing",
                    "url": "https://example.com/ai-news-1",
                    "source_type": "web",
                    "published_date": "2025-08-15",
                },
            ),
            Document(
                page_content="Python continues to be the most popular programming language for machine learning and data science projects worldwide.",
                metadata={
                    "source": "https://example.com/python-trends",
                    "title": "Python Dominates Machine Learning Development",
                    "url": "https://example.com/python-trends",
                    "source_type": "web",
                    "published_date": "2025-08-14",
                },
            ),
            Document(
                page_content="Vector databases are becoming essential infrastructure for AI applications requiring semantic search capabilities.",
                metadata={
                    "source": "https://example.com/vector-db-guide",
                    "title": "The Rise of Vector Databases in AI",
                    "url": "https://example.com/vector-db-guide",
                    "source_type": "web",
                    "published_date": "2025-08-13",
                },
            ),
            Document(
                page_content="Large language models are transforming how businesses approach customer service and content generation.",
                metadata={
                    "source": "https://example.com/llm-business",
                    "title": "LLMs Revolutionize Business Operations",
                    "url": "https://example.com/llm-business",
                    "source_type": "web",
                    "published_date": "2025-08-12",
                },
            ),
            Document(
                page_content="Retrieval Augmented Generation (RAG) systems combine the best of search and generation for accurate AI responses.",
                metadata={
                    "source": "https://example.com/rag-systems",
                    "title": "Understanding RAG Architecture",
                    "url": "https://example.com/rag-systems",
                    "source_type": "web",
                    "published_date": "2025-08-11",
                },
            ),
        ]

    def search(self, query: str, k: int = 3, **kwargs) -> list[Document]:
        """
        Mock web search with simple keyword matching.

        Args:
            query: Search query
            k: Maximum number of results
            **kwargs: Additional search parameters

        Returns:
            List of mock search results
        """
        # Simulate network delay
        if self.delay > 0:
            time.sleep(self.delay)

        # Simple keyword-based filtering for more realistic behavior
        query_words = set(query.lower().split())
        scored_results = []

        for doc in self.mock_results:
            content_words = set(doc.page_content.lower().split())
            title_words = set(doc.metadata.get("title", "").lower().split())

            # Weight title matches higher than content matches
            content_overlap = len(query_words.intersection(content_words))
            title_overlap = len(query_words.intersection(title_words))

            score = content_overlap + (title_overlap * 2)  # Title matches worth 2x
            scored_results.append((doc, score))

        # Sort by relevance score and return top k
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Add some randomness to make it more realistic
        results = [doc for doc, _ in scored_results[:k]]

        # Simulate slight randomization in results order
        if len(results) > 1 and random.random() > 0.7:
            # Occasionally swap the top 2 results
            results[0], results[1] = results[1], results[0]

        return results

    def add_mock_result(self, document: Document) -> None:
        """Add a mock search result."""
        self.mock_results.append(document)

    def clear_results(self) -> None:
        """Clear all mock results."""
        self.mock_results.clear()

    def set_results(self, results: list[Document]) -> None:
        """Replace all mock results."""
        self.mock_results = results
