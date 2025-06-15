import logging
from logging.config import dictConfig

from utils.config import Config, Environment

debug_level_env = [Environment.GIT_HUB, Environment.LOCAL]

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s [%(levelname)s]: %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S"},
        "json": {
            "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG" if Config.ENVIRONMENT in debug_level_env else "INFO",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "json",
            "filename": "app.log",
            "level": "DEBUG" if Config.ENVIRONMENT in debug_level_env else "INFO",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "app": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "root": {"handlers": ["console"], "level": "WARNING"},
}

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("app")
