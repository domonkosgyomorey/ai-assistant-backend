from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud.storage import Client


class GCPHelper:
    def __init__(self, bucket_name: str, prefix: str = ""):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.client = Client()
        self.bucket = self.client.bucket(bucket_name)

    def _download_single_blob(self, blob, destination_dir: Path, preserve_structure: bool = True) -> None:
        """Helper method to download a single blob."""
        if blob.name.endswith("/"):
            return

        if preserve_structure:
            local_file_path = destination_dir / blob.name
        else:
            if self.prefix and blob.name.startswith(self.prefix):
                relative_name = blob.name[len(self.prefix) :].lstrip("/")
            else:
                relative_name = Path(blob.name).name
            local_file_path = destination_dir / relative_name

        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(local_file_path))

    def download_all(self, destination_dir: str, max_workers: int = 10, preserve_structure: bool = True):
        """
        Download all files from the bucket using concurrent threads.

        Args:
            destination_dir (str): Local directory to save all files
            max_workers (int): Maximum number of concurrent download threads
            preserve_structure (bool): If True, preserves the folder structure from the bucket.
                                     If False, flattens files into the destination directory.
        """
        destination = Path(destination_dir)
        destination.mkdir(parents=True, exist_ok=True)

        blobs = list(self.client.list_blobs(self.bucket, prefix=self.prefix))
        total_files = len([b for b in blobs if not b.name.endswith("/")])

        print(f"Starting download of {total_files} files with {max_workers} concurrent threads...")
        if self.prefix:
            print(f"Downloading from subfolder: {self.prefix}")
        if not preserve_structure:
            print("Flattening folder structure in local destination")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_blob = {
                executor.submit(self._download_single_blob, blob, destination, preserve_structure): blob
                for blob in blobs
            }

            completed = 0
            for future in as_completed(future_to_blob):
                future.result()
                completed += 1
                if completed % 50 == 0:
                    print(f"Progress: {completed}/{total_files} files downloaded")

        print(f"All {total_files} files downloaded to: {destination}")

    def upload_file(self, local_file_path: str, destination_blob_name: str | None = None) -> None:
        """
        Upload a single file to the bucket.

        Args:
            local_file_path (str): Path to the local file to upload
            destination_blob_name (str): Name for the file in the bucket. If None, uses the filename.
                                       Will be prefixed with the subfolder if one is specified.

        """
        local_path = Path(local_file_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        if destination_blob_name is None:
            destination_blob_name = local_path.name

        if self.prefix:
            destination_blob_name = f"{self.prefix.rstrip('/')}/{destination_blob_name}"

        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(str(local_path))

    def download_file(self, blob_name: str, destination_path: str | None = None) -> None:
        """
        Download a single specific file from the bucket.

        Args:
            blob_name (str): Name of the file in the bucket to download
            destination_path (str): Local path where to save the file.
                                  If None, uses the blob name in current directory.

        Raises:
            FileNotFoundError: If the blob doesn't exist in the bucket
        """
        if self.prefix:
            full_blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        else:
            full_blob_name = blob_name

        blob = self.bucket.blob(full_blob_name)

        if not blob.exists():
            raise FileNotFoundError(f"File not found in bucket: {full_blob_name}")

        if destination_path is None:
            destination_path = Path(blob_name).name

        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(destination))
