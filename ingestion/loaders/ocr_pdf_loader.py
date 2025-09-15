import os
import re
from io import BytesIO
from typing import List

import fitz
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from ingestion.interfaces import DocumentLoader
from ingestion.utils.common import CustomDocument


class OCRPDFLoader(DocumentLoader):
    def __init__(self, ocr_first: bool = True, improve_image: bool = True, min_confidence: int = 30):
        """
        OCR-focused PDF loader for better text extraction from scanned documents.

        Args:
            ocr_first: Use OCR as primary method instead of fallback
            improve_image: Apply image enhancement before OCR
            min_confidence: Minimum OCR confidence threshold (0-100)
        """
        self.ocr_first = ocr_first
        self.improve_image = improve_image
        self.min_confidence = min_confidence

    def load(self, path: str) -> List[CustomDocument]:
        """Load and process PDF with OCR-first approach."""
        doc = fitz.open(path)
        title = doc.metadata.get("title", "") or os.path.splitext(os.path.basename(path))[0]
        description = doc.metadata.get("description", "") or ""
        page_count = doc.page_count

        pages_docs = []
        for i in range(page_count):
            page = doc.load_page(i)

            if self.ocr_first:
                # OCR-first approach
                content = self._extract_with_ocr_primary(page)
                if not content.strip():
                    # Fallback to regular text extraction
                    content = page.get_text("text")
            else:
                # Traditional approach with better OCR fallback
                content = page.get_text("text")
                if len(content.strip()) < 100:  # More generous threshold
                    ocr_content = self._extract_with_ocr_primary(page)
                    if ocr_content.strip():
                        content = ocr_content

            # Clean and format content
            content = self._clean_text(content)

            if content.strip():
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

    def _extract_with_ocr_primary(self, page) -> str:
        """Enhanced OCR extraction with image preprocessing."""
        try:
            # Convert page to high-resolution image
            mat = fitz.Matrix(3.0, 3.0)  # 3x scale for better OCR accuracy
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))

            # Apply image enhancements if enabled
            if self.improve_image:
                img = self._enhance_image_for_ocr(img)

            # Multiple OCR attempts with different configurations
            ocr_configs = [
                "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}\"'-/\\@#$%^&*+=<>|~ ",
                "--psm 4 --oem 3",
                "--psm 6 --oem 1",
                "--psm 3",
            ]

            best_text = ""
            best_confidence = 0

            for config in ocr_configs:
                try:
                    # Get OCR data with confidence scores
                    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

                    # Calculate average confidence
                    confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)

                        if avg_confidence > best_confidence and avg_confidence >= self.min_confidence:
                            text = pytesseract.image_to_string(img, config=config)
                            if len(text.strip()) > len(best_text.strip()):
                                best_text = text
                                best_confidence = avg_confidence
                except Exception:
                    continue

            return best_text

        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""

    def _enhance_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Apply image enhancements to improve OCR accuracy."""
        try:
            # Convert to grayscale for better OCR
            if img.mode != "L":
                img = img.convert("L")

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)

            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)

            # Apply slight blur to reduce noise, then sharpen
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

            # Increase brightness slightly
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)

            return img

        except Exception:
            return img  # Return original if enhancement fails

    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning for OCR output."""
        if not text:
            return ""

        # Fix common OCR errors
        ocr_corrections = {
            r"\b0\b": "O",  # Zero to O
            r"\b1\b(?=\w)": "I",  # 1 to I when followed by letters
            r"\bl\b": "I",  # lowercase l to I
            r"\brn\b": "m",  # rn to m
            r"\bvv\b": "w",  # vv to w
            r"\b5\b(?=[a-zA-Z])": "S",  # 5 to S before letters
        }

        for pattern, replacement in ocr_corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Remove excessive whitespace and artifacts
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # Standalone numbers (page nums)
        text = re.sub(r"^[^\w\s]*$", "", text, flags=re.MULTILINE)  # Symbol-only lines

        # Fix sentence spacing
        text = re.sub(r"([.!?])\s*\n\s*([A-Z])", r"\1 \2", text)
        text = re.sub(r"([a-z])\s*\n\s*([a-z])", r"\1 \2", text)

        # Remove duplicate consecutive words
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)

        # Clean up spacing
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    def supports_file(self, path: str) -> bool:
        """Check if file is a supported PDF."""
        return path.lower().endswith(".pdf")

    def get_loader_info(self) -> dict:
        """Return information about loader capabilities."""
        return {
            "name": "OCRPDFLoader",
            "supports": ["pdf"],
            "features": {
                "ocr_first": self.ocr_first,
                "image_enhancement": self.improve_image,
                "confidence_filtering": True,
                "multi_config_ocr": True,
                "advanced_cleaning": True,
            },
        }
