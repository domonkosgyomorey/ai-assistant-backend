import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class SplitterConfig:
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200


@dataclass
class MongoConfig:
    URI: str = os.environ["MONGO_URI"]
    DB_NAME: str = "assistant"
    COLLECTION_NAME: str = "knowledge"
    VECTOR_SRACH_INDEX_NAME: str = "embedding"


@dataclass
class GCPConfig:
    BUCKET_NAME: str = "ai-assistant-dev-docs"
    PREFIX: str = ""


@dataclass
class Config:
    splitter: SplitterConfig = SplitterConfig()
    mongo: MongoConfig = MongoConfig()
    gcp: GCPConfig = GCPConfig()


config = Config()
