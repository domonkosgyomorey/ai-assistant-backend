from abc import ABC, abstractmethod

from utils.common_types import CustomDocument


class ClusterDescriptor(ABC):
    """
    Description: An abstract base class for describing clusters of documents. This base class provides a strict interface for all cluster descriptor implementations.
    """

    @abstractmethod
    def describe(self, clustered_docs: dict[str, list[CustomDocument]]) -> dict[str, str]:
        """
        Description: Abstract method to name clusters of documents.
        Args:
            clustered_docs (dict[str, list[CustomDocument]]): A dictionary where keys are cluster identifiers and values are lists of documents belonging to that cluster.

        Returns:
            dict[str, str]: A dictionary where keys are cluster identifiers and values are the names/descriptions of those clusters.
        """
        pass
