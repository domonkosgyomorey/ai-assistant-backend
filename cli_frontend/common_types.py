from enum import StrEnum


class COMMANDS(StrEnum):
    EXIT = "exit"
    SAVE_CONV = "save"


class MessageType(StrEnum):
    REQUEST = "request"
    RESPONSE = "response"
