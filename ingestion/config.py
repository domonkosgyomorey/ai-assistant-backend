import os
from enum import StrEnum

from ingestion.utils.gcp_secret import get_secret

CWD = os.path.abspath(os.getcwd())


class ENVIRONMENT(StrEnum):
    LOCAL = "local"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    GITHUB = "github"


class MongoConfig:
    URI: str = get_secret("MONGO_URI")
    DB_NAME: str = "assistant"
    COLLECTION_NAME: str = "knowledge"
    VECTOR_SRACH_INDEX_NAME: str = "embedding"
    RELEVANCE_SCORE_FN: str = "cosine"


class GCPConfig:
    BUCKET_NAME: str = "ai-assistant-dev-docs"
    PREFIX: str = ""


class Tavily:
    API_KEY = get_secret("TAVILY_API_KEY")


class Config:
    mongo: MongoConfig = MongoConfig()
    gcp: GCPConfig = GCPConfig()
    llm: str = "gemini-2.0"
    embedding: str = "gemini-embedding-001"
    tavily: Tavily = Tavily()
    environment: ENVIRONMENT = ENVIRONMENT.LOCAL


config = Config()
