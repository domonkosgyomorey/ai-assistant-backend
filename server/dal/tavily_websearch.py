from core.config.config import config
from langchain_core.documents import Document
from langchain_core.retrievers import RetrieverLike
from tavily import TavilyClient


class TavilyRetriever(RetrieverLike):
    def __init__(self, include_domains: list[str] | None = None, exclude_domains: list[str] | None = None):
        self.client = TavilyClient(
            api_key=config.tavily.API_KEY, include_domains=include_domains, exclude_domains=exclude_domains
        )

    def retrieve(self, query: str, k: int = 3, **kwargs) -> list[Document]:
        response = self.client.search(query=query, max_results=k, **kwargs)
        results = response.get("results", [])
        docs = []
        for item in results:
            content = item.get("content", "")
            url = item.get("url", "")
            title = item.get("title", "")
            metadata = {"source": url, "title": title}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs
