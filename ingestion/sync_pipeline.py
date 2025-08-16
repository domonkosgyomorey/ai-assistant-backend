import os
import sys

sys.path.append(os.path.join(os.getcwd(), os.path.abspath(__file__)))

import tempfile
from argparse import ArgumentParser

from ingestion.config import config
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.processors.document_processor import DocumentProcessorImpl
from ingestion.storage_fetchers.gcp_bucket_fetcher import GCPBucketFetcher
from ingestion.stores.mongo_store import MongoAtlasVectorStore
from ingestion.utils.logger import logger


class SyncPipeline:
    def __init__(self, bucket_name: str, collection: str, use_existing_collection: bool, clear_collection_before: bool):
        self.fetcher = GCPBucketFetcher(bucket_name)
        self.store = MongoAtlasVectorStore(
            collection_name=collection,
            use_existing_collection=use_existing_collection,
            clear_collection_before=clear_collection_before,
        )
        self.loader = PDFLoader()
        self.processor = DocumentProcessorImpl()

    def sync(self):
        known_keys = self.store.list_source_keys()
        new_keys = self.fetcher.new_files(known_keys)
        logger.info(f"Currently {len(known_keys)} files in the collection.")
        logger.info(f"{len(new_keys)} will be synced from GCP bucket {self.fetcher.bucket.name}.")
        for key in new_keys:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = self.fetcher.fetch_file(key, f"{tmpdir}/{key.split('/')[-1]}")
                docs = self.loader.load(local_path)
                chunks = self.processor.process(docs)
                self.store.save(chunks)

        logger.info(f"Synced {len(new_keys)} new files.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Sync Pipeline for GCP Bucket to MongoDB Atlas")
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

    SyncPipeline(
        bucket_name=config.gcp.BUCKET_NAME,
        collection=config.mongo.COLLECTION_NAME,
        use_existing_collection=args.use_existing_collection,
        clear_collection_before=args.clear_collection_before,
    ).sync()
