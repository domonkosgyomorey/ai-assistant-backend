from pydantic import BaseModel


class RelevanceDecision(BaseModel):
    """Result of deciding if retrieved docs are relevant to the user query."""

    relevant: bool
    reason: str
