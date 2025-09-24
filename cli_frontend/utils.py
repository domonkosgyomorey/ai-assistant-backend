from datetime import datetime
from typing import Dict, List


def create_message_payload(messages: List[Dict[str, str]]) -> dict:
    """Create payload with properly formatted messages for the new RequestModel structure."""
    if not messages:
        raise ValueError("Messages list cannot be empty")

    return {
        "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
        "chatId": "default",
    }


def get_time() -> str:
    return datetime.now().isoformat(sep=" ")


def num_of_bytes(string: str) -> int:
    return len(string.encode("utf-8"))
