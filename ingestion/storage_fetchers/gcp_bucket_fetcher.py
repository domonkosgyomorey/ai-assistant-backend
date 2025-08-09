from pathlib import Path

from google.cloud.storage import Client

from ingestion.interfaces import Fetcher


class GCPBucketFetcher(Fetcher):
    def __init__(self, bucket_name: str, prefix: str = ""):
        self.client = Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix

    def list_files(self) -> list[str]:
        return [blob.name for blob in self.client.list_blobs(self.bucket, prefix=self.prefix)]

    def fetch_file(self, key: str, dest_path: str) -> Path:
        blob = self.bucket.blob(key)
        casted_dest_path = Path(dest_path)
        casted_dest_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(casted_dest_path)
        return casted_dest_path

    def new_files(self, known_keys: set[str]) -> list[str]:
        return [f for f in self.list_files() if f not in known_keys]
