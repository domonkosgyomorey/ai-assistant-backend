import os
from typing import List

import fitz

from ingestion.interfaces import DocumentLoader
from ingestion.utils.common import CustomDocument


class PDFLoader(DocumentLoader):
    def load(self, path: str) -> List[CustomDocument]:
        doc = fitz.open(path)
        title = doc.metadata.get("title", "") or ""
        description = doc.metadata.get("description", "") or ""
        page_count = doc.page_count

        pages_docs = []
        for i in range(page_count):
            page = doc.load_page(i)
            content = page.get_text("text")
            pages_docs.append(
                CustomDocument(
                    content=content,
                    file_path=os.path.basename(path),
                    title=title,
                    page_number=i + 1,
                    page_count=page_count,
                    short_description=description,
                )
            )

        return pages_docs
