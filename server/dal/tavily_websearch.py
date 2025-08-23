from core.config.config import config
from core.interfaces import BaseRetriever
from langchain_core.documents import Document
from tavily import TavilyClient


class TavilyRetriever(BaseRetriever):
    def __init__(
        self,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        relevance_tolerance: float = 0.4,
    ):
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.client = TavilyClient(api_key=config.tavily.API_KEY)
        self.relevance_tolerance = relevance_tolerance

    def retrieve(self, query: str, k: int = 3, **kwargs) -> list[Document]:
        response = self.client.search(
            query=query,
            max_results=k,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            **kwargs,
        )
        results = response.get("results", [])
        docs: list[Document] = []
        for item in results:
            content = item.get("content", "")
            url = item.get("url", "")
            title = item.get("title", "")
            score = item.get("score", 0)
            metadata = {"source": url, "title": title, "score": score}
            if score > self.relevance_tolerance:
                docs.append(Document(page_content=content, metadata=metadata))

        return docs
