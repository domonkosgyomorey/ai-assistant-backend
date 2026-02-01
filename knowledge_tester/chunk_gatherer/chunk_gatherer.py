from abc import ABC, abstractmethod


class ChunkGatherer(ABC):
    @abstractmethod
    def gather(self, key: str) -> str:
        pass
