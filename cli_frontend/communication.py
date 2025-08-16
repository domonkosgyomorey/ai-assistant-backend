import json
from collections import deque

import requests
from colorama import Fore
from common_types import MessageType
from config import Config
from safe_utils import safe_open
from utils import create_simple_payload, get_time, num_of_bytes


class Communication:
    def __init__(self, conv_hist_size: int = 3):
        self.conv_hist_size = conv_hist_size
        self.history: deque = deque(maxlen=self.conv_hist_size)
        self.conversations: dict = {"messages": []}

    def send_message(self, message: str) -> str:
        """Send a message to the server and handle streaming response with detailed output."""
        # Record the request
        self.conversations["messages"].append({"text": message, "type": str(MessageType.REQUEST), "time": get_time()})

        payload = create_simple_payload(message)

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

            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)
                    full_response += chunk

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
