from typing import Any, List

from core.config import config
from langchain_core.documents import Document
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import RunnableConfig
from tavily import TavilyClient


class TavilyRetriever(RetrieverLike):
    def __init__(self, k: int = 5, search_depth: str = "basic"):
        self.client = TavilyClient(api_key=config.tavily.API_KEY)
        self.k = k
        self.search_depth = search_depth

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        response = self.client.search(query=query, max_results=self.k, search_depth=self.search_depth, **kwargs)
        results = response.get("results", [])
        docs = []
        for item in results:
            content = item.get("content", "")
            url = item.get("url", "")
            title = item.get("title", "")
            metadata = {"source": url, "title": title}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def invoke(self, input: str, config: RunnableConfig | None = None, **kwargs: Any) -> list[Document]:
        return self._get_relevant_documents(input)
