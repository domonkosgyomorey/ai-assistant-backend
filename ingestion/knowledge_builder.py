import argparse
import os

from langchain.schema import Document

from ingestion.interfaces.interfaces import DocumentLoader, DocumentProcessor
from ingestion.loaders.ocr_pdf_loader import OCRPDFLoader
from ingestion.processors.document_processor import DocumentProcessorImpl
from ingestion.utils.common import CustomDocument
from ingestion.utils.logger import logger


def get_all_files(directory):
    all_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            all_files.append(full_path)
    return all_files


class KnowledgeBuilder:
    def __init__(self, loader: DocumentLoader, processor: DocumentProcessor):
        self.loader: DocumentLoader = loader
        self.processor: DocumentProcessor = processor

    def run(self, source_folder: str):
        files = get_all_files(source_folder)
        logger.info(f"Found {len(files)} files in {source_folder} and {len(files)} new files to process.")

        for file_path in files:
            logger.info(f"Processing file: {file_path}")
            raw_docs: list[CustomDocument] = self.loader.load(file_path)
            processed_docs: list[Document] = self.processor.process(raw_docs)
            for doc in processed_docs:
                doc.page_content = f"File name: {file_path} | File title: {doc.metadata.get('title', 'None')} | Content in page {doc.metadata.get('page_number', 0)}: {doc.page_content}"
            with open("asd.json", "w", encoding="utf-8") as f:
                f.write("[\n" + ",\n".join([f'"{doc.page_content}"' for doc in processed_docs]) + "\n]")
            # self.store.save(processed_docs)
        logger.info("Pipeline run completed for all files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Builder Pipeline")
    parser.add_argument("--source-folder", required=True, help="Folder containing source documents to ingest")
    parser.add_argument("--collection-name", required=False, help="MongoDB collection name (overrides config)")
    parser.add_argument(
        "--use-existing-collection", action="store_true", default=False, help="Fail if collection already exists"
    )
    parser.add_argument(
        "--clear-collection-before",
        action="store_true",
        default=False,
        help="Clear collection before running the pipeline",
    )

    args = parser.parse_args()

    loader: DocumentLoader = OCRPDFLoader()
    processor: DocumentProcessor = DocumentProcessorImpl()
    # store: DocumentStore = MongoAtlasVectorStore(
    #    collection_name=args.collection_name,
    #    use_existing_collection=args.use_existing_collection,
    #    clear_collection_before=args.clear_collection_before,
    # )

    pipeline = KnowledgeBuilder(loader, processor)
    pipeline.run(args.source_folder)
