import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessorImpl:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            chunk_size=800,
            chunk_overlap=200,
        )

    def process(self, docs: list[Document]) -> list[Document]:
        """Process and split documents into smaller chunks, returning a List of Documents."""
        if not docs:
            logger.warning("No documents provided for processing.")
            return []

        logger.info(f"Processing {len(docs)} documents.")

        documents = [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in docs
        ]

        return self.splitter.split_documents(documents)
