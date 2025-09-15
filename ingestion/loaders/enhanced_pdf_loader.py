import os
import re
from io import BytesIO
from typing import List

import fitz
import pytesseract
from PIL import Image

from ingestion.interfaces import DocumentLoader
from ingestion.utils.common import CustomDocument


class EnhancedPDFLoader(DocumentLoader):
    def __init__(self, use_ocr: bool = True, extract_tables: bool = True):
        """
        Enhanced PDF loader with OCR and table extraction capabilities.

        Args:
            use_ocr: Enable OCR for scanned documents
            extract_tables: Enable table detection and extraction
        """
        self.use_ocr = use_ocr
        self.extract_tables = extract_tables

    def load(self, path: str) -> List[CustomDocument]:
        """Load and process PDF with enhanced extraction capabilities."""
        doc = fitz.open(path)
        title = doc.metadata.get("title", "") or os.path.splitext(os.path.basename(path))[0]
        description = doc.metadata.get("description", "") or ""
        page_count = doc.page_count

        pages_docs = []
        for i in range(page_count):
            page = doc.load_page(i)

            # Extract text content
            content = self._extract_page_content(page)

            # Clean and format content
            content = self._clean_text(content)

            if content.strip():  # Only add pages with content
                pages_docs.append(
                    CustomDocument(
                        content=content,
                        source=os.path.basename(path),
                        title=title,
                        page_number=i + 1,
                        page_count=page_count,
                        short_description=description,
                    )
                )

        doc.close()
        return pages_docs

    def _extract_page_content(self, page) -> str:
        """Extract content from a single page using multiple methods."""
        content_parts = []

        # 1. Extract regular text
        text = page.get_text("text")
        if text.strip():
            content_parts.append(text)

        # 2. Extract tables if enabled
        if self.extract_tables:
            tables = self._extract_tables(page)
            content_parts.extend(tables)

        # 3. Use OCR for images/scanned content if text is minimal
        if self.use_ocr and len(text.strip()) < 50:
            ocr_text = self._extract_with_ocr(page)
            if ocr_text.strip():
                content_parts.append(f"\n[OCR Content]\n{ocr_text}")

        return "\n\n".join(content_parts)

    def _extract_tables(self, page) -> List[str]:
        """Extract tables from page and convert to text format."""
        tables = []
        try:
            # Find table-like structures using text blocks
            blocks = page.get_text("dict")["blocks"]
            table_candidates = []

            for block in blocks:
                if "lines" in block:
                    lines = block["lines"]
                    if len(lines) > 2:  # Potential table with multiple rows
                        table_text = []
                        for line in lines:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"] + " "
                            table_text.append(line_text.strip())

                        # Check if it looks like a table (has consistent spacing/alignment)
                        if self._is_table_like(table_text):
                            table_candidates.append("\n".join(table_text))

            # Format table candidates
            for i, table in enumerate(table_candidates):
                tables.append(f"\n[Table {i + 1}]\n{table}")

        except Exception:
            pass  # Skip table extraction on error

        return tables

    def _is_table_like(self, lines: List[str]) -> bool:
        """Simple heuristic to detect table-like content."""
        if len(lines) < 3:
            return False

        # Check for consistent column-like spacing
        tab_counts = [line.count("\t") for line in lines]
        space_patterns = [len(re.findall(r"\s{3,}", line)) for line in lines]

        # If most lines have similar tab/space patterns, likely a table
        return (max(tab_counts) > 0 and len(set(tab_counts)) <= 2) or (
            max(space_patterns) > 1 and len(set(space_patterns)) <= 3
        )

    def _extract_with_ocr(self, page) -> str:
        """Extract text using OCR for scanned content."""
        try:
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))

            # Apply OCR
            ocr_text = pytesseract.image_to_string(img, config="--psm 6")
            return ocr_text
        except Exception:
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        # Remove common PDF artifacts
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # Page numbers
        text = re.sub(r"^[^\w\s]*$", "", text, flags=re.MULTILINE)  # Symbol-only lines

        # Fix common OCR errors
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)  # Duplicate words
        text = re.sub(r"([.!?])\s*\n\s*([a-z])", r"\1 \2", text)  # Fix sentence breaks

        # Normalize spacing
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    def supports_file(self, path: str) -> bool:
        """Check if file is a supported PDF."""
        return path.lower().endswith(".pdf")

    def get_loader_info(self) -> dict:
        """Return information about loader capabilities."""
        return {
            "name": "EnhancedPDFLoader",
            "supports": ["pdf"],
            "features": {
                "ocr": self.use_ocr,
                "tables": self.extract_tables,
                "text_cleaning": True,
                "metadata_extraction": True,
            },
        }
