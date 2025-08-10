from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from langchain.schema import Document

from ingestion.utils.common import CustomDocument


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[CustomDocument]:
        """Load documents from file and return list of langchain Documents."""


class DocumentProcessor(ABC):
    @abstractmethod
    def process(self, docs: list[CustomDocument]) -> list[Document]:
        """Process and split documents into smaller chunks, returning a list of Documents."""


class DocumentStore(ABC):
    @abstractmethod
    def save(self, docs: list[Document]):
        """Save documents to storage backend."""

    @abstractmethod
    def list_source_keys(self) -> list[str]:
        """list all source keys (file identifiers) in the store."""


class Fetcher(Protocol):
    def list_files(self) -> list[str]:
        """Return list of file identifiers (paths, keys, etc.)"""
        ...

    def fetch_file(self, key: str, dest_path: str) -> Path:
        """Download a file to a local destination"""
        ...

    def new_files(self, known_keys: set[str]) -> list[str]:
        """Return files not in the known set"""
        ...
