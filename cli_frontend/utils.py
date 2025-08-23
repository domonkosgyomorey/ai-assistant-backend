from datetime import datetime
from typing import Dict, List


def create_message_payload(messages: List[Dict[str, str]]) -> dict:
    """Create payload with properly formatted messages."""
    return {"messages": messages}


def create_simple_payload(message: str) -> dict:
    # Deprecated: Use create_message_payload instead
    return {"question": message}


def get_time() -> str:
    return datetime.now().isoformat(sep=" ")


def num_of_bytes(string: str) -> int:
    return len(string.encode("utf-8"))
