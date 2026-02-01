import os

from resource_loader.resource_loader import ResourceLoader
from utils.common_types import CustomDocument


class TxtLoader(ResourceLoader):
    @classmethod
    def load(cls, path: str) -> list[CustomDocument]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        total_characters = len(content)
        # Heuristic: assume 1800 characters per page
        estimated_page_size = 1800
        total_pages = max(
            1, total_characters // estimated_page_size + (1 if total_characters % estimated_page_size else 0)
        )

        documents = []

        for page in range(total_pages):
            start_idx = page * estimated_page_size
            end_idx = start_idx + estimated_page_size
            page_content = content[start_idx:end_idx].strip()
            if page_content:
                documents.append(
                    CustomDocument(
                        content=page_content,
                        page_number=page + 1,
                        file_name=os.path.basename(path),
                        source_type="file",
                    )
                )
        return documents
