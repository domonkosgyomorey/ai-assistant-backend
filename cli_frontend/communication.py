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

        # Must end with user message
        if messages[-1]["role"] != "user":
            return False

        # Check alternating pattern
        for i in range(1, len(messages)):
            current_role = messages[i]["role"]
            previous_role = messages[i - 1]["role"]

            # Messages should alternate
            if current_role == previous_role:
                return False

        return True

    def send_message(self, message: str) -> str:
        """Send a message to the server and handle streaming response with detailed output."""
        # Record the request
        self.conversations["messages"].append({"text": message, "type": str(MessageType.REQUEST), "time": get_time()})

        # Convert conversation history to the expected message format
        # Ensure we maintain proper alternating order: user -> assistant -> user -> assistant...
        messages = []
        for msg in self.conversations["messages"]:
            if msg["type"] == str(MessageType.REQUEST):
                messages.append({"role": "user", "content": msg["text"]})
            elif msg["type"] == str(MessageType.RESPONSE):
                messages.append({"role": "assistant", "content": msg["text"]})

        # Ensure the sequence ends with user message (which it should since we just added one)
        if not self._validate_message_sequence(messages):
            raise ValueError(
                "Invalid message sequence: messages must alternate between user and assistant and end with user message"
            )

        payload = create_message_payload(messages)

        # Display request information
        print(f"{Fore.BLUE}📤 Sending request to: {Fore.CYAN}{Config.ENDPOINT}{Fore.RESET}")
        print(f"{Fore.BLUE}📝 Payload: {Fore.WHITE}{json.dumps(payload, indent=2)}{Fore.RESET}")
        print(f"{Fore.BLUE}📊 Request size: {Fore.YELLOW}{num_of_bytes(json.dumps(payload))} bytes{Fore.RESET}")
        print(f"{Fore.GREEN}🔄 Streaming response:{Fore.RESET}\n")

        try:
            # Make streaming request
            response = requests.post(Config.ENDPOINT, json=payload, stream=True)
            response.raise_for_status()

            # Handle streaming response
            full_response = ""
            print(f"[{Fore.GREEN}AI{Fore.RESET}]: ", end="", flush=True)

            # Use response.iter_content with proper text handling
            response.encoding = "utf-8"  # Ensure proper encoding
            for chunk in response.iter_content(chunk_size=1, decode_unicode=False):
                if chunk:
                    # Decode bytes to string if necessary
                    if isinstance(chunk, bytes):
                        try:
                            chunk_str = chunk.decode("utf-8")
                        except UnicodeDecodeError:
                            chunk_str = chunk.decode("utf-8", errors="ignore")
                    else:
                        chunk_str = str(chunk)

                    print(chunk_str, end="", flush=True)
                    full_response += chunk_str

            print()  # New line after streaming is complete

            # Record the response
            self.conversations["messages"].append(
                {"text": full_response, "type": str(MessageType.RESPONSE), "time": get_time()}
            )

            # Display final statistics
            print(f"\n{Fore.BLUE}📈 Response size: {Fore.YELLOW}{num_of_bytes(full_response)} bytes{Fore.RESET}")

            return full_response

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
