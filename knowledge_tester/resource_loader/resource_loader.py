from abc import ABC, abstractmethod

from utils.common_types import CustomDocument


class ResourceLoader(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> list[CustomDocument]:
        pass
