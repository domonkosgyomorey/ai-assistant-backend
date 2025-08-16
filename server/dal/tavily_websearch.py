from typing import Any, List

from core.config.config import config
from core.interfaces import WebSearchRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import RunnableConfig
from tavily import TavilyClient


class TavilyRetriever(RetrieverLike, WebSearchRetriever):
    """Tavily Web Search implementation of WebSearchRetriever interface."""

    def __init__(self, k: int = 5, search_depth: str = "basic"):
        self.client = TavilyClient(api_key=config.tavily.API_KEY)
        self.k = k
        self.search_depth = search_depth

    def search(self, query: str, k: int = 3, **kwargs) -> list[Document]:
        """Implement WebSearchRetriever interface method."""
        response = self.client.search(query=query, max_results=k, search_depth=self.search_depth, **kwargs)
        results = response.get("results", [])
        docs = []
        for item in results:
            content = item.get("content", "")
            url = item.get("url", "")
            title = item.get("title", "")
            metadata = {"source": url, "title": title}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    # Keep LangChain RetrieverLike compatibility
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self.search(query, k=self.k, **kwargs)

    def invoke(self, input: str, config: RunnableConfig | None = None, **kwargs: Any) -> list[Document]:
        return self.search(input, k=self.k, **kwargs)
