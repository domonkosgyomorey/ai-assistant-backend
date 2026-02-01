import uuid
import warnings

import numpy as np
from cluster_descriptor.interfaces import ClusterDescriptor
from document_clustering.comparative_clustering import compare_pipelines
from document_clustering.document_clustering import Cluster, ClusteringAlgorithm, EmbeddedDocument

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class HDBSCANDocumentClustering(ClusteringAlgorithm):
    @classmethod
    def cluster(
        cls,
        docs: list[EmbeddedDocument],
        cluster_descriptor: ClusterDescriptor,
    ) -> list[Cluster]:
        embeddings = np.array([doc["embedding"] for doc in docs])
        card = len(embeddings)

        if card < 5:
            single_cluster = Cluster(
                id=str(uuid.uuid4()),
                documents=[doc["document"] for doc in docs],
                description="",
            )
            return [single_cluster]

        # Always run comparative pipeline analysis (UMAP vs PCA+UMAP)
        results = compare_pipelines(
            docs=docs,
            cluster_descriptor=cluster_descriptor,
            base_output_dir="pipeline_comparison",
            pca_components=150,
        )
        return results["umap"]["clusters"]
