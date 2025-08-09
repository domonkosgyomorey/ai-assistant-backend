import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SplitterConfig:
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))


@dataclass
class MongoConfig:
    URI: str = os.getenv(
        "MONGO_URI",
        "mongodb+srv://domonkosgyomorey:PsKvBxxIIArkd2zX@cluster0.qvlbn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    )
    DB_NAME: str = os.getenv("MONGO_DB_NAME", "ai-assistant")
    COLLECTION_NAME: str = os.getenv("MONGO_COLLECTION", "knowledge")
    VECTOR_SRACH_INDEX_NAME: str = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "embedding")


@dataclass
class GCPConfig:
    BUCKET_NAME: str = os.getenv("GCP_BUCKET_NAME", "ai-assistant-dev-docs")
    CREDENTIALS_PATH: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)
    PREFIX: str = os.getenv("GCP_PREFIX", "")


@dataclass
class Config:
    splitter: SplitterConfig = SplitterConfig()
    mongo: MongoConfig = MongoConfig()
    gcp: GCPConfig = GCPConfig()


config = Config()
