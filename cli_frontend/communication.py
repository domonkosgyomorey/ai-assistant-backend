import json
from collections import deque
from http import HTTPStatus

import requests
from common_types import MessageType
from config import Config
from requests import HTTPError
from safe_utils import safe_open
from utils import create_simple_payload, get_time


class Communication:
    def __init__(self, conv_hist_size: int = 3):
        self.conv_hist_size = conv_hist_size
        self.history: deque = deque(maxlen=self.conv_hist_size)
        self.conversations: dict = {"messages": []}

    def send_message(self, message: str) -> str:
        raw_response = requests.post(Config.ENDPOINT, json=create_simple_payload(message))
        self.conversations["messages"].append({"text": message, "type": str(MessageType.REQUEST), "time": get_time()})

        if raw_response.status_code == HTTPStatus.OK:
            response: dict = raw_response.json()
            answer = str(response.get("response", "Empty"))
            self.conversations["messages"].append(
                {"text": answer, "type": str(MessageType.RESPONSE), "time": get_time()}
            )
            return answer
        elif raw_response.status_code == HTTPStatus.BAD_REQUEST:
            raise HTTPError("Bad request: the request cannot be processed.")
        else:
            raise HTTPError(f"Unknown error: {raw_response.status_code} - {raw_response.text}")

    def save_conversion(self):
        with safe_open(Config.CONV_SAVE_FILE, mode="w", encoding="utf-8") as f:
            json.dump(self.conversations, f, ensure_ascii=False)
