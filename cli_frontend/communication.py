import json
from collections import deque

import requests
from colorama import Fore
from common_types import MessageType
from config import Config
from safe_utils import safe_open
from utils import create_message_payload, get_time, num_of_bytes


class Communication:
    def __init__(self, conv_hist_size: int = 3):
        self.conv_hist_size = conv_hist_size
        self.history: deque = deque(maxlen=self.conv_hist_size)
        self.conversations: dict = {"messages": []}

    def _validate_message_sequence(self, messages: list) -> bool:
        """Validate that messages alternate between user and assistant and end with user message."""
        if not messages:
            return False

        if messages[-1]["role"] != "user":
            return False

        for i in range(1, len(messages)):
            current_role = messages[i]["role"]
            previous_role = messages[i - 1]["role"]

            if current_role == previous_role:
                return False

        return True

    def send_message(self, message: str) -> str:
        """Send a message to the server and handle JSON response with detailed output."""
        self.conversations["messages"].append({"text": message, "type": str(MessageType.REQUEST), "time": get_time()})

        messages = []
        for msg in self.conversations["messages"]:
            if msg["type"] == str(MessageType.REQUEST):
                messages.append({"role": "user", "content": msg["text"]})
            elif msg["type"] == str(MessageType.RESPONSE):
                messages.append({"role": "ai", "content": msg["text"]})

        if not self._validate_message_sequence(messages):
            raise ValueError(
                "Invalid message sequence: messages must alternate between user and assistant and end with user message"
            )

        payload = create_message_payload(messages)

        print(f"{Fore.BLUE}📤 Sending request to: {Fore.CYAN}{Config.ENDPOINT}{Fore.RESET}")
        print(f"{Fore.BLUE}📝 Payload: {Fore.WHITE}{json.dumps(payload, indent=2)}{Fore.RESET}")
        print(f"{Fore.BLUE}📊 Request size: {Fore.YELLOW}{num_of_bytes(json.dumps(payload))} bytes{Fore.RESET}")
        print(f"{Fore.GREEN}🔄 Processing request:{Fore.RESET}\n")

        try:
            response = requests.post(Config.ENDPOINT, json=payload)
            response.raise_for_status()

            response_data = response.json()

            ai_content = response_data.get("message", {}).get("content", "")
            metadata = response_data.get("metadata", {})

            print(f"[{Fore.GREEN}AI{Fore.RESET}]: {ai_content}")

            if metadata:
                print(f"{Fore.BLUE}📋 Metadata: {Fore.WHITE}{json.dumps(metadata, indent=2)}{Fore.RESET}")

            self.conversations["messages"].append(
                {"text": ai_content, "type": str(MessageType.RESPONSE), "time": get_time()}
            )

            print(f"\n{Fore.BLUE}📈 Response size: {Fore.YELLOW}{num_of_bytes(ai_content)} bytes{Fore.RESET}")

            return ai_content

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"{Fore.RED}❌ {error_msg}{Fore.RESET}")
            if hasattr(e, "response") and e.response is not None:
                print(f"{Fore.RED}Status code: {e.response.status_code}{Fore.RESET}")
                print(f"{Fore.RED}Response: {e.response.text}{Fore.RESET}")
            return error_msg

    def save_conversion(self):
        with safe_open(Config.CONV_SAVE_FILE, mode="w", encoding="utf-8") as f:
            json.dump(self.conversations, f, ensure_ascii=False)
