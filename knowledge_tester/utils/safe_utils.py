import os
from pathlib import Path

from werkzeug.security import safe_join as werkzeug_safe_join

BASE_DIR = os.getcwd()


def safe_join(base_dir: str | os.PathLike, *paths: str | os.PathLike) -> str:
    base_path = Path(base_dir).resolve()
    normalized_parts: list[str] = []

    for path in paths:
        path_obj = Path(path)
        if path_obj.is_absolute():
            resolved = path_obj.resolve()
            try:
                relative = resolved.relative_to(base_path)
            except ValueError as exc:
                raise ValueError(f"Unsafe path: {paths} cannot be joined with base directory: {base_dir}") from exc
            normalized_parts.append(relative.as_posix())
        else:
            normalized_parts.append(path_obj.as_posix())

    safe_path = werkzeug_safe_join(base_path.as_posix(), *normalized_parts)
    if safe_path is not None:
        return str(Path(safe_path))
    raise ValueError(f"Unsafe path: {paths} cannot be joined with base directory: {base_dir}")


def safe_open(*paths: str | os.PathLike, **kwargs):
    """
    Open a file safely by joining the base directory with the provided paths.
    """
    safe_path = safe_join(BASE_DIR, *paths)
    return open(safe_path, **kwargs)


def safe_walk(*paths: str | os.PathLike):
    """
    Walk through a directory safely by joining the base directory with the provided paths.
    """
    safe_path = safe_join(BASE_DIR, *paths)
    return os.walk(safe_path)
