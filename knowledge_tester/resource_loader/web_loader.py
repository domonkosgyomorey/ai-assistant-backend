import os
import sys

from bs4 import BeautifulSoup
from resource_loader.resource_loader import CustomDocument, ResourceLoader
from tavily import TavilyClient

sys.path.append(os.path.join(os.getcwd(), "server"))
from core.config.config import Config

client = TavilyClient(api_key=Config.tavily.API_KEY)


class WebLoader(ResourceLoader):
    @classmethod
    def load(cls, path: str) -> list[CustomDocument]:
        scrape_result = client.extract(path)
        raw_html = scrape_result.get("results", [{}])[0].get("raw_content", "")
        soup = BeautifulSoup(raw_html, "html.parser")
        clean_text = soup.get_text(separator="\n", strip=True)
        doc = CustomDocument(
            content=clean_text,
            page_number=None,
            file_name=path,
            source_type="url",
        )
        return [doc]
