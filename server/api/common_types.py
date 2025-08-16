from pydantic import BaseModel, Field


class RequestModel(BaseModel):
    question: str = Field(description="The question to ask the knowledge agent")
