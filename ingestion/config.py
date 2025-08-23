from ingestion.utils.gcp_secret import get_secret


class SplitterConfig:
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 500


class MongoConfig:
    URI: str = get_secret("MONGO_URI")
    DB_NAME: str = "assistant"
    COLLECTION_NAME: str = "knowledge"
    VECTOR_SEARCH_INDEX_NAME: str = "embedding"


class GCPConfig:
    BUCKET_NAME: str = "ai-assistant-dev-docs"
    PREFIX: str = ""


class Config:
    splitter: SplitterConfig = SplitterConfig()
    mongo: MongoConfig = MongoConfig()
    gcp: GCPConfig = GCPConfig()
    embedding: str = "gemini-embedding-001"


config = Config()
