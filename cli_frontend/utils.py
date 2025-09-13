from datetime import datetime
from typing import Dict, List


def create_message_payload(messages: List[Dict[str, str]]) -> dict:
    """Create payload with properly formatted messages for the new RequestModel structure."""
    if not messages:
        raise ValueError("Messages list cannot be empty")

    # The last message should be the current user message
    current_message = messages[-1]

    # All previous messages are context (should end with AI message if any context exists)
    context = messages[:-1] if len(messages) > 1 else []

    return {
        "message": {"role": current_message["role"], "content": current_message["content"]},
        "context": [{"role": msg["role"], "content": msg["content"]} for msg in context],
        "chatId": "default",
    }


def get_time() -> str:
    return datetime.now().isoformat(sep=" ")


def num_of_bytes(string: str) -> int:
    return len(string.encode("utf-8"))
