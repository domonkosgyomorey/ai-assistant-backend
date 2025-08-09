import os

from dotenv import load_dotenv

load_dotenv()


class SplitterConfig:
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200


class MongoConfig:
    URI: str = os.environ["MONGO_URI"]
    DB_NAME: str = "assistant"
    COLLECTION_NAME: str = "knowledge"
    VECTOR_SRACH_INDEX_NAME: str = "embedding"


class GCPConfig:
    BUCKET_NAME: str = "ai-assistant-dev-docs"
    PREFIX: str = ""


class Config:
    splitter: SplitterConfig = SplitterConfig()
    mongo: MongoConfig = MongoConfig()
    gcp: GCPConfig = GCPConfig()


config = Config()
