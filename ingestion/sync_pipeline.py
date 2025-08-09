import tempfile

from ingestion.config import config
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.processors.document_processor import DocumentProcessorImpl
from ingestion.storage_fetchers.gcp_bucket_fetcher import GCPBucketFetcher
from ingestion.stores.mongo_store import MongoAtlasVectorStore
from ingestion.utils.logger import logger


class SyncPipeline:
    def __init__(self, bucket_name: str, collection: str):
        self.fetcher = GCPBucketFetcher(bucket_name)
        self.store = MongoAtlasVectorStore(collection)
        self.loader = PDFLoader()
        self.processor = DocumentProcessorImpl()

    def sync(self):
        known_keys = self.store.list_source_keys()
        new_keys = self.fetcher.new_files(known_keys)

        for key in new_keys:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = self.fetcher.fetch_file(key, f"{tmpdir}/{key.split('/')[-1]}")
                docs = self.loader.load(local_path)
                chunks = self.processor.process(docs)
                self.store.add_documents(chunks)

        logger.info(f"Synced {len(new_keys)} new files.")


if __name__ == "__main__":
    SyncPipeline(bucket_name=config.gcp.BUCKET_NAME, collection=config.mongo.COLLECTION_NAME).sync()
