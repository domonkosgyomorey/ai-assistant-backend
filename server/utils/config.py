import os
from enum import StrEnum

from utils.safe_join import CWD, safe_join


class Environment(StrEnum):
    PRODUCTION = "production"
    GIT_HUB = "github"
    LOCAL = "local"


class Config:
    PROJECT_NAME = "AI Assistant Backend"
    ENVIRONMENT: Environment = Environment(os.environ.get("env", "local"))
    LLM_FILE_NAME = "openchat_3.5.Q4_K_M.gguf"
    LLM_PATH = safe_join(CWD, "models", LLM_FILE_NAME)

    DB_NAME = "ai-assistant"
    COLLECTION_NAME = "knowledge"
    MONGODB_CONNECTION_STRING = "mongodb+srv://domonkosgyomorey:FRgXFKOV2mNaAf7t@cluster0.qvlbn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
