from typing import List, Optional

from langchain_core.documents import Document

from ingestion.config import config
from ingestion.storage_fetchers.gcp_public_uploader import GCPPublicUploader


class PublicDocumentHelper:
    def __init__(self):
        self.public_uploader = GCPPublicUploader(config.gcp.PUBLIC_BUCKET_NAME)

    def get_document_public_url(self, source_key: str) -> Optional[str]:
        """
        Get the public URL for a document if it exists.

        Args:
            source_key: Original source key (e.g., 'pdfs/student_guide.pdf')

        Returns:
            Public URL if document exists, None otherwise
        """
        if self.public_uploader.pdf_exists(source_key):
            return self.public_uploader.get_public_url(source_key)
        return None

    def find_public_url_by_filename(self, filename: str) -> Optional[str]:
        """
        Find public URL by filename.

        Args:
            filename: Just the filename (e.g., 'student_guide.pdf')

        Returns:
            Public URL if found, None otherwise
        """
        if self.public_uploader.pdf_exists(filename):
            return self.public_uploader.get_public_url(filename)

        return None

    def extract_public_url_from_document(self, document: Document) -> Optional[str]:
        """Extract public URL from document metadata or generate it from source."""
        stored_url = document.metadata.get("public_url")
        if stored_url:
            return stored_url

        source = document.metadata.get("source")
        if source:
            return self.find_public_url_by_filename(source)

        return None

    def is_document_viewable(self, document: Document) -> bool:
        """Check if document has a viewable public URL."""
        return document.metadata.get("viewable", False) and document.metadata.get("public_url") is not None

    def get_viewable_source_info(self, document: Document) -> dict:
        """
        Get complete viewing information for a document.

        Returns:
            Dict with source info including public URL if available
        """
        source_info = {
            "source": document.metadata.get("source", "Unknown"),
            "page": document.metadata.get("page_number"),
            "viewable": self.is_document_viewable(document),
            "public_url": self.extract_public_url_from_document(document),
        }

        return source_info

    def list_all_public_documents(self) -> List[str]:
        """List all documents available in the public bucket."""
        return self.public_uploader.list_public_pdfs()


def get_document_viewer_link(document: Document) -> Optional[str]:
    """Quick function to get a viewer link for a document."""
    helper = PublicDocumentHelper()
    return helper.extract_public_url_from_document(document)


def format_source_with_link(document: Document) -> str:
    """
    Format a source reference with a clickable link if available.

    Returns:
        Formatted string like "Source: document.pdf (Page 5) - [View PDF](url)" or just "Source: document.pdf (Page 5)"
    """
    helper = PublicDocumentHelper()
    source_info = helper.get_viewable_source_info(document)

    source_name = source_info["source"]
    page = source_info["page"]
    public_url = source_info["public_url"]

    source_text = f"{source_name}"

    if page:
        source_text += f"#page={page}"

    if public_url:
        source_text = f"[{source_text}]({public_url})"

    return source_text


def get_sources_with_links(documents: List[Document]) -> List[dict]:
    """
    Get a list of sources with their public links for multiple documents.

    Returns:
        List of dicts with source information and links
    """
    helper = PublicDocumentHelper()
    sources = []

    for doc in documents:
        source_info = helper.get_viewable_source_info(doc)
        sources.append(source_info)

    return sources
