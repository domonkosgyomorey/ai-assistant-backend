from typing import List

from google.cloud.storage import Client

from ingestion.utils.logger import logger


class GCPPublicUploader:
    """Uploads PDFs to a public GCP bucket for viewing from AI Assistant."""

    def __init__(self, bucket_name: str):
        self.client = Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name

    def ensure_bucket_is_public(self):
        """Ensure the bucket has public read access."""
        try:
            # Make bucket publicly readable
            policy = self.bucket.get_iam_policy(requested_policy_version=3)
            policy.bindings.append({"role": "roles/storage.objectViewer", "members": {"allUsers"}})
            self.bucket.set_iam_policy(policy)
            logger.info(f"Bucket {self.bucket_name} is now publicly readable")
        except Exception as e:
            logger.warning(f"Could not set public access for bucket {self.bucket_name}: {e}")
            logger.info("Please ensure the bucket has public read access manually")

    def upload_pdf(self, local_pdf_path: str, source_key: str) -> str:
        """
        Upload a PDF to the public bucket.

        Args:
            local_pdf_path: Local path to the PDF file
            source_key: Original source key (e.g., 'pdfs/student_guide.pdf')

        Returns:
            str: Public URL to access the PDF
        """
        # Use the original filename for public access
        blob_name = source_key

        # Upload the PDF
        blob = self.bucket.blob(blob_name)

        # Set content type for PDFs
        blob.upload_from_filename(local_pdf_path, content_type="application/pdf")

        # Make this specific blob publicly readable
        blob.make_public()

        # Generate public URL
        public_url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"

        logger.info(f"Uploaded PDF to public bucket: {public_url}")
        return public_url

    def get_public_url(self, source_key: str) -> str:
        """Get the public URL for a specific document."""
        return f"https://storage.googleapis.com/{self.bucket_name}/{source_key}"

    def list_public_pdfs(self) -> List[str]:
        """List all PDFs in the public bucket."""
        blobs = self.client.list_blobs(self.bucket)
        return [blob.name for blob in blobs if blob.name.lower().endswith(".pdf")]

    def pdf_exists(self, source_key: str) -> bool:
        """Check if a PDF already exists in the public bucket."""
        blob = self.bucket.blob(source_key)
        return blob.exists()

    def delete_pdf(self, source_key: str) -> bool:
        """Delete a PDF from the public bucket."""
        try:
            blob = self.bucket.blob(source_key)
            blob.delete()
            logger.info(f"Deleted PDF from public bucket: {source_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete PDF {source_key}: {e}")
            return False
