from datetime import datetime


def create_simple_payload(message: str) -> dict:
    return {"message": message}


def get_time() -> str:
    return datetime.now().isoformat(sep=" ")


def num_of_bytes(string: str) -> int:
    return len(string.encode("utf-8"))
