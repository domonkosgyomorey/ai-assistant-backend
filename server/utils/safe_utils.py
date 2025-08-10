import os

from pathlib2 import PurePosixPath

project_dir = os.path.abspath(os.getcwd())


def _safe_join(base_dir: str, *paths: str) -> str:
    return str(PurePosixPath(base_dir, *paths))


def safe_open(*paths: str, mode: str, encoding: str):
    path = _safe_join(project_dir, *paths)
    return open(path, mode=mode, encoding=encoding)
