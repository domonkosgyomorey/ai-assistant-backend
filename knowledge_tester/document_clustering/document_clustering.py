from abc import ABC, abstractmethod
from typing import TypedDict

from cluster_descriptor.interfaces import ClusterDescriptor
from utils.common_types import Cluster, CustomDocument


class EmbeddedDocument(TypedDict):
    document: CustomDocument
    embedding: list[float]


class ClusteringAlgorithm(ABC):
    @classmethod
    @abstractmethod
    def cluster(cls, config: list[EmbeddedDocument], cluster_descriptor: ClusterDescriptor) -> list[Cluster]:
        pass
