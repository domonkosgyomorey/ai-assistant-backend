from abc import ABC, abstractmethod
from typing import TypedDict


class InteractionMessage(TypedDict):
    prompt: str


class InteractionGrounding(TypedDict):
    type: str
    source: str


class InteractionResponse(TypedDict):
    response: str
    groundings: list[InteractionGrounding]


class Interaction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def send_message(self, message: InteractionMessage) -> InteractionResponse:
        pass
