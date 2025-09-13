from typing import List, Literal, Union

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator


class MessageModel(BaseModel):
    """Represents a single message in the conversation."""

    role: Literal["user", "ai"] = Field(description="The role of the message sender")
    content: str = Field(description="The content of the message")

    def to_langchain_message(self) -> Union[HumanMessage, AIMessage]:
        """Convert to LangChain message type."""
        if self.role == "user":
            return HumanMessage(content=self.content)
        elif self.role == "ai":
            return AIMessage(content=self.content)
        else:
            raise ValueError(f"Unknown role: {self.role}")


class RequestModel(BaseModel):
    context: List[MessageModel] = Field(description="The conversation messages to send to the knowledge agent")
    chatId: str = Field(default="default", description="The chat session identifier")
    message: MessageModel = Field(description="The latest message from the user")

    @field_validator("context")
    @classmethod
    def validate_message_sequence(cls, context: List[MessageModel]) -> List[MessageModel]:
        """Validate that messages alternate between user and assistant and end with user message."""
        if not context:
            raise ValueError("Messages list cannot be empty")

        if context[-1].role != "ai":
            raise ValueError("Message sequence must end with a user message")

        for i in range(1, len(context)):
            current_role = context[i].role
            previous_role = context[i - 1].role

            if current_role == previous_role:
                raise ValueError(
                    f"Messages must alternate between user and assistant. Found consecutive {current_role} messages at positions {i - 1} and {i}"
                )

        if len(context) > 1 and context[0].role != "user":
            raise ValueError("First message in a conversation should be from user")

        return context
