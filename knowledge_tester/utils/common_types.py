from typing import Literal, Optional, TypedDict

from langchain.schema.document import Document
from pydantic import BaseModel, Field, field_validator, model_validator


class CustomDocument(TypedDict):
    content: str
    page_number: int | None
    file_name: str
    source_type: str


def custom_document_to_document(custom_doc: CustomDocument) -> Document:
    """Convert CustomDocument to LangChain Document."""
    return Document(
        page_content=custom_doc["content"],
        metadata={
            "page_number": custom_doc["page_number"],
            "file_name": custom_doc["file_name"],
            "source_type": custom_doc["source_type"],
        },
    )


def document_to_custom_document(doc: Document) -> CustomDocument:
    """Convert LangChain Document to CustomDocument."""
    return CustomDocument(
        content=doc.page_content,
        page_number=doc.metadata.get("page_number"),
        file_name=doc.metadata.get("file_name", ""),
        source_type=doc.metadata.get("source_type", ""),
    )


class Cluster(TypedDict):
    id: str
    description: str
    documents: list[CustomDocument]
    source: str


class ReferenceKeywords(BaseModel):
    required: list[str] = Field(default_factory=list, description="Keywords that must appear in the answer")
    blocked: list[str] = Field(default_factory=list, description="Keywords that must not appear in the answer")


SourceType = Literal["file", "url", "web", "text", "video"]


class References(BaseModel):
    answer: Optional[str] = Field(default=None, description="Reference answer text")
    chunks: Optional[str] = Field(default=None, description="Supporting text excerpts for the reference answer")
    sources: Optional[dict[str, SourceType]] = Field(default=None, description="Mapping of source names to their type")
    keywords: Optional[ReferenceKeywords] = Field(
        default=None, description="Keywords that must or must not appear in the answer"
    )

    @model_validator(mode="after")
    def check_at_least_one_field(self):
        if not self.model_fields_set:
            raise ValueError("At least one field must be provided")
        return self


class GenerationMetadata(BaseModel):
    cluster_description: Optional[str] = Field(default=None, description="Cluster summary or topic description")


class TestCase(BaseModel):
    question_type: Literal["general", "specific", "situational"] = Field(
        default="general", description="Type of question"
    )
    prompt: str = Field(..., description="The user question or model input")
    references: References = Field(..., description="References to validate the system")
    generation_metadata: Optional[GenerationMetadata] = Field(
        default=None, description="Metadata about the generation process"
    )

    @field_validator("prompt")
    @classmethod
    def check_prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace")
        return v


class TestCaseWithGroups(TestCase):
    groups: Optional[list[str]] = Field(default=None, description="List of groups the test case belongs to")


class SystemData(BaseModel):
    answer: Optional[str] = Field(default=None, description="Reference answer text")
    chunks: Optional[str] = Field(default=None, description="Supporting text excerpts for the reference answer")
    sources: Optional[dict[str, SourceType]] = Field(default=None, description="Mapping of source names to their type")


class AnsweredTestCase(TestCaseWithGroups):
    system: SystemData = Field(..., description="System-generated answer and related data")


class JudgedTestCase(AnsweredTestCase):
    judgement: dict[str, dict] = Field(..., description="Judgements from various evaluators")
