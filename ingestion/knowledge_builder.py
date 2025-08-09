import argparse
import os

from langchain.schema import Document

from ingestion.interfaces.interfaces import DocumentLoader, DocumentProcessor, DocumentStore
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.processors.document_processor import DocumentProcessorImpl
from ingestion.stores.mongo_store import MongoAtlasVectorStore
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
    def __init__(self, loader: DocumentLoader, processor: DocumentProcessor, store: DocumentStore):
        self.loader: DocumentLoader = loader
        self.processor: DocumentProcessor = processor
        self.store: DocumentStore = store

    def run(self, source_folder: str):
        files = get_all_files(source_folder)
        files_in_db = self.store.list_source_keys()
        new_files = [file for file in files if file not in files_in_db]
        logger.info(f"Found {len(files)} files in {source_folder} and {len(new_files)} new files to process.")

        for file_path in new_files:
            logger.info(f"Processing file: {file_path}")
            raw_docs: list[CustomDocument] = self.loader.load(file_path)
            processed_docs: list[Document] = self.processor.process(raw_docs)
            self.store.save(processed_docs)
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

    loader: DocumentLoader = PDFLoader()
    processor: DocumentProcessor = DocumentProcessorImpl()
    store: DocumentStore = MongoAtlasVectorStore(
        collection_name=args.collection_name,
        fail_on_collection_exists=args.fail_on_collection_exists,
        clear_collection_before=args.clear_collection_before,
    )

    pipeline = KnowledgeBuilder(loader, processor, store)
    pipeline.run(args.source_folder)
