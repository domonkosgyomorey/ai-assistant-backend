import pickle
from typing import List

from google.cloud.storage import Client
from langchain_core.documents import Document

from ingestion.utils.logger import logger


class GCPDocumentUploader:
    """Uploads LangChain Document objects to GCP bucket for evaluation purposes."""

    def __init__(self, bucket_name: str):
        self.client = Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload_document(self, document: Document, document_id: str) -> str:
        """
        Upload a single document to GCP bucket using document ID as filename.

        Args:
            document: LangChain Document object
            document_id: Unique document ID to use as filename

        Returns:
            str: GCP path where document was uploaded
        """
        blob_key = f"{document_id}.pkl"

        blob = self.bucket.blob(blob_key)
        pickle_data = pickle.dumps(document)
        blob.upload_from_string(pickle_data, content_type="application/octet-stream")

        logger.info(f"Uploaded document to: gs://{self.bucket.name}/{blob_key}")
        return blob_key

    def upload_documents(self, documents: List[Document]) -> List[str]:
        """
        Upload multiple documents using their metadata ID or auto-generated ID.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List[str]: List of GCP paths where documents were uploaded
        """
        uploaded_paths = []

        for i, doc in enumerate(documents):
            document_id = doc.metadata.get("_id") or doc.metadata.get("id") or f"doc_{i}"
            path = self.upload_document(doc, document_id)
            uploaded_paths.append(path)

        logger.info(f"Uploaded {len(documents)} documents to GCP bucket: {self.bucket.name}")
        return uploaded_paths

    def download_document(self, document_id: str) -> Document:
        """Download and reconstruct a Document object from GCP bucket."""
        blob_key = f"{document_id}.pkl"
        blob = self.bucket.blob(blob_key)

        if not blob.exists():
            raise FileNotFoundError(f"Document {document_id} not found in bucket")

        pickle_data = blob.download_as_bytes()
        return pickle.loads(pickle_data)

    def download_documents(self, document_ids: List[str]) -> List[Document]:
        """Download multiple documents by their IDs."""
        documents = []
        for doc_id in document_ids:
            try:
                doc = self.download_document(doc_id)
                documents.append(doc)
            except FileNotFoundError:
                logger.warning(f"Document {doc_id} not found, skipping")

        return documents

    def list_document_ids(self) -> List[str]:
        """List all document IDs in the bucket."""
        blobs = self.client.list_blobs(self.bucket)
        return [blob.name.replace(".pkl", "") for blob in blobs if blob.name.endswith(".pkl")]

    def clear_bucket(self) -> bool:
        """Delete all documents from the evaluation bucket using efficient bulk operations."""
        try:
            # Get all blobs at once without downloading content
            blobs = list(self.bucket.list_blobs())
            if not blobs:
                logger.info(f"Evaluation bucket {self.bucket.name} is already empty")
                return True

            logger.info(f"Clearing {len(blobs)} files from evaluation bucket {self.bucket.name}")

            # Use delete_blobs for efficient bulk deletion
            # This is much faster than individual deletions or batching
            self.bucket.delete_blobs(blobs)

            logger.info(f"Successfully cleared evaluation bucket {self.bucket.name} ({len(blobs)} files deleted)")
            return True
        except Exception as e:
            logger.error(f"Failed to clear evaluation bucket {self.bucket.name}: {e}")
            return False

    def delete_document(self, document_id: str) -> bool:
        """Delete a specific document from the bucket."""
        try:
            blob_key = f"{document_id}.pkl"
            blob = self.bucket.blob(blob_key)
            blob.delete()
            logger.info(f"Deleted document from evaluation bucket: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
