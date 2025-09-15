import os
import sys

sys.path.append(os.path.join(os.getcwd(), os.path.abspath(__file__)))

import tempfile
import uuid
from argparse import ArgumentParser

from langchain_core.documents import Document

from ingestion.config import config
from ingestion.loaders.ocr_pdf_loader import OCRPDFLoader
from ingestion.processors.document_processor import DocumentProcessorImpl
from ingestion.storage_fetchers.gcp_bucket_fetcher import GCPBucketFetcher
from ingestion.storage_fetchers.gcp_document_uploader import GCPDocumentUploader
from ingestion.storage_fetchers.gcp_public_uploader import GCPPublicUploader
from ingestion.stores.mongo_store import MongoAtlasVectorStore
from ingestion.utils.logger import logger


class SyncPipeline:
    def __init__(
        self,
        bucket_name: str,
        collection: str,
        use_existing_collection: bool,
        clear_collection_before: bool,
        upload_for_evaluation: bool = False,
        upload_to_public: bool = True,
        clear_evaluation_bucket: bool = False,
        clear_public_bucket: bool = False,
    ):
        self.fetcher = GCPBucketFetcher(bucket_name)
        self.store = MongoAtlasVectorStore(
            collection_name=collection,
            use_existing_collection=use_existing_collection,
            clear_collection_before=clear_collection_before,
        )
        self.loader = OCRPDFLoader()
        self.processor = DocumentProcessorImpl()

        self.upload_for_evaluation = upload_for_evaluation
        self.clear_evaluation_bucket = clear_evaluation_bucket
        if self.upload_for_evaluation:
            self.document_uploader = GCPDocumentUploader(config.gcp.EVALUATION_BUCKET_NAME)
        else:
            self.document_uploader = None

        self.upload_to_public = upload_to_public
        self.clear_public_bucket = clear_public_bucket
        if self.upload_to_public:
            self.public_uploader = GCPPublicUploader(config.gcp.PUBLIC_BUCKET_NAME)
            self.public_uploader.ensure_bucket_is_public()
        else:
            self.public_uploader = None

    def sync(self):
        # Clear buckets if requested
        if self.clear_evaluation_bucket and self.document_uploader:
            logger.info("Clearing evaluation bucket before sync...")
            if self.document_uploader.clear_bucket():
                logger.info("✓ Evaluation bucket cleared successfully")
            else:
                logger.error("✗ Failed to clear evaluation bucket")

        if self.clear_public_bucket and self.public_uploader:
            logger.info("Clearing public bucket before sync...")
            if self.public_uploader.clear_bucket():
                logger.info("✓ Public bucket cleared successfully")
            else:
                logger.error("✗ Failed to clear public bucket")

        known_keys = self.store.list_source_keys()
        new_keys = self.fetcher.new_files(known_keys)
        logger.info(f"Currently {len(known_keys)} files in the collection.")
        logger.info(f"{len(new_keys)} will be synced from GCP bucket {self.fetcher.bucket.name}.")

        if self.upload_for_evaluation:
            logger.info("Evaluation mode enabled - documents will be uploaded to GCP bucket.")

        if self.upload_to_public:
            logger.info("Public upload enabled - PDFs will be uploaded to public bucket.")

        for key in new_keys:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = self.fetcher.fetch_file(key, f"{tmpdir}/{key.split('/')[-1]}")

                public_url = None
                if self.upload_to_public and self.public_uploader:
                    try:
                        if not self.public_uploader.pdf_exists(key):
                            public_url = self.public_uploader.upload_pdf(str(local_path), key)
                            logger.info(f"Uploaded PDF to public bucket: {key}")
                        else:
                            public_url = self.public_uploader.get_public_url(key)
                            logger.info(f"PDF already exists in public bucket: {key}")
                    except Exception as e:
                        logger.error(f"Failed to upload PDF to public bucket: {key}. Error: {e}")

                docs = self.loader.load(local_path)
                chunks = self.processor.process(docs)

                if public_url:
                    for chunk in chunks:
                        chunk.metadata["public_url"] = public_url
                        chunk.metadata["viewable"] = True

                self.store.save(chunks)

                if self.upload_for_evaluation and self.document_uploader:
                    try:
                        base_filename = key.split("/")[-1].replace(".pdf", "").replace(".", "_")

                        langchain_docs = []
                        for doc_idx, doc in enumerate(docs):
                            unique_id = f"{base_filename}_doc_{doc_idx}_{uuid.uuid4().hex[:8]}"

                            langchain_doc = Document(
                                page_content=doc.content,
                                metadata={
                                    "id": unique_id,
                                    "source": doc.source,
                                    "title": doc.title,
                                    "short_description": doc.short_description,
                                    "page_number": doc.page_number,
                                    "page_count": doc.page_count,
                                },
                            )
                            langchain_docs.append(langchain_doc)

                        self.document_uploader.upload_documents(langchain_docs)
                        logger.info(f"Successfully uploaded {len(langchain_docs)} documents for evaluation: {key}")
                    except Exception as e:
                        logger.error(f"Failed to upload documents for evaluation: {key}. Error: {e}")

        logger.info(f"Synced {len(new_keys)} new files.")

        if self.upload_for_evaluation:
            logger.info(f"Evaluation documents uploaded to bucket: {config.gcp.EVALUATION_BUCKET_NAME}")

        if self.upload_to_public:
            logger.info(f"PDFs uploaded to public bucket: {config.gcp.PUBLIC_BUCKET_NAME}")


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
    parser.add_argument(
        "--upload-for-evaluation",
        action="store_true",
        default=False,
        help="Upload processed documents to GCP bucket for evaluation purposes",
    )
    parser.add_argument(
        "--no-public-upload",
        action="store_true",
        default=False,
        help="Disable uploading PDFs to public bucket (enabled by default)",
    )
    parser.add_argument(
        "--clear-evaluation-bucket",
        action="store_true",
        default=False,
        help="Clear evaluation bucket before uploading new documents",
    )
    parser.add_argument(
        "--clear-public-bucket",
        action="store_true",
        default=False,
        help="Clear public bucket before uploading new PDFs",
    )
    args = parser.parse_args()

    SyncPipeline(
        bucket_name=config.gcp.BUCKET_NAME,
        collection=config.mongo.COLLECTION_NAME,
        use_existing_collection=args.use_existing_collection,
        clear_collection_before=args.clear_collection_before,
        upload_for_evaluation=args.upload_for_evaluation,
        upload_to_public=not args.no_public_upload,
        clear_evaluation_bucket=args.clear_evaluation_bucket,
        clear_public_bucket=args.clear_public_bucket,
    ).sync()
