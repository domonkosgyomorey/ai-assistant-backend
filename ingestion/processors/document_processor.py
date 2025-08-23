import logging

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ingestion.config import config
from ingestion.interfaces import DocumentProcessor
from ingestion.utils.common import CustomDocument

logger = logging.getLogger(__name__)


class DocumentProcessorImpl(DocumentProcessor):
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.splitter.CHUNK_SIZE,
            chunk_overlap=config.splitter.CHUNK_OVERLAP,
        )

    def process(self, docs: list[CustomDocument]) -> list[Document]:
        """Process and split documents into smaller chunks, returning a List of Documents."""
        if not docs:
            logger.warning("No documents provided for processing.")
            return []

        logger.info(f"Processing {len(docs)} documents.")
        documents = [
            Document(
                page_content=doc.content,
                metadata={
                    "source": doc.source,
                    "title": doc.title,
                    "page_number": doc.page_number,
                    "page_count": doc.page_count,
                    "description": doc.short_description,
                },
            )
            for doc in docs
        ]

        return self.splitter.split_documents(documents)
