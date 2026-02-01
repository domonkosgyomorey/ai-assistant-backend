import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from resource_loader.resource_loader import ResourceLoader
from utils.common_types import CustomDocument

from ingestion.loaders.ocr_pdf_loader import OCRPDFLoader


class PDFLoader(ResourceLoader):
    """PDF loader that uses OCR for better text extraction from scanned documents."""

    def __init__(self, ocr_first: bool = True, improve_image: bool = True, min_confidence: int = 30):
        """
        Initialize PDF loader with OCR capabilities.

        Args:
            ocr_first: Use OCR as primary method instead of fallback
            improve_image: Apply image enhancement before OCR
            min_confidence: Minimum OCR confidence threshold (0-100)
        """
        self.ocr_loader = OCRPDFLoader(ocr_first=ocr_first, improve_image=improve_image, min_confidence=min_confidence)

    @classmethod
    def load(cls, path: str) -> list[CustomDocument]:
        """Load PDF with OCR support."""
        loader = cls()
        return loader._load_with_ocr(path)

    def _load_with_ocr(self, path: str) -> list[CustomDocument]:
        """Load PDF using the OCR loader from ingestion."""
        ingestion_docs = self.ocr_loader.load(path)

        custom_docs = []
        for doc in ingestion_docs:
            custom_doc = CustomDocument(
                content=doc.content,
                page_number=doc.page_number,
                file_name=os.path.basename(path),
                source_type="file",
            )
            custom_docs.append(custom_doc)

        return custom_docs
