import os
from enum import StrEnum

from core.utils.gcp_secret import get_secret
from core.utils.safe_utils import _safe_join

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


class LLMConfig:
    LLM_FILE_NAME = "openchat_3.5.Q4_K_M.gguf"
    LLM_PATH = _safe_join(CWD, "models", LLM_FILE_NAME)


class Tavily:
    API_KEY = get_secret("TAVILY_API_KEY")


class Config:
    mongo: MongoConfig = MongoConfig()
    gcp: GCPConfig = GCPConfig()
    llm: LLMConfig = LLMConfig()
    tavily: Tavily = Tavily()
    environment: ENVIRONMENT = ENVIRONMENT.LOCAL


config = Config()
