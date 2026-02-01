import requests
from system_interaction.interaction import Interaction, InteractionMessage, InteractionResponse


def create_payload(
    messages: list[dict],
) -> dict:
    payload = {
        "messages": [
            {
                "role": msg["type"],
                "content": msg["text"],
            }
            for msg in messages
        ],
        "chatId": "local chat",
    }
    return payload


def create_simple_payload(text: str) -> dict:
    return create_payload([{"type": "user", "text": text}])


class LocalInteraction(Interaction):
    def __init__(self, base_url: str = "http://localhost:8081", endpoint: str = "/prompt"):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}{endpoint}"

    def send_message(self, message: InteractionMessage) -> InteractionResponse:
        payload = create_simple_payload(message["prompt"])

        response = requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        groundings = []
        for doc_id in data["metadata"].get("db_docs_ids", []):
            groundings.append({"source": doc_id, "type": "file"})
        for link in data["metadata"].get("web_links", []):
            groundings.append({"source": link, "type": "web"})

        return {
            "response": data["message"]["content"],
            "groundings": groundings,
        }
