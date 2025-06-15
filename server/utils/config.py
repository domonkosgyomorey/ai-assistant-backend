import os
from enum import StrEnum


class Environment(StrEnum):
    PRODUCTION = "production"
    GIT_HUB = "github"
    LOCAL = "local"


class Config:
    PROJECT_NAME = "AI Assistant Backend"
    ENVIRONMENT: Environment = Environment(os.environ.get("env", "local"))
