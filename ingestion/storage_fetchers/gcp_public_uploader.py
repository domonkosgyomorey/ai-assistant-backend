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
        """
        Note: With uniform bucket-level access enabled, public access is managed
        via IAM policies in Terraform, not through bucket ACLs.
        This method is kept for compatibility but does nothing.
        """
        logger.info(f"Bucket {self.bucket_name} public access is managed via Terraform IAM policies")
        return

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

        # Note: No need to call blob.make_public() because uniform bucket-level access
        # is enabled and the bucket already has public read access via IAM

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

    def clear_bucket(self) -> bool:
        """Delete all files from the public bucket using efficient bulk operations."""
        try:
            # Get all blobs at once without downloading content
            blobs = list(self.bucket.list_blobs())
            if not blobs:
                logger.info(f"Public bucket {self.bucket_name} is already empty")
                return True

            logger.info(f"Clearing {len(blobs)} files from public bucket {self.bucket_name}")

            # Use delete_blobs for efficient bulk deletion
            # This is much faster than individual deletions
            blob_names = [blob.name for blob in blobs]
            self.bucket.delete_blobs(blobs)

            logger.info(f"Successfully cleared public bucket {self.bucket_name} ({len(blob_names)} files deleted)")
            return True
        except Exception as e:
            logger.error(f"Failed to clear public bucket {self.bucket_name}: {e}")
            return False
