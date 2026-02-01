import json
import math
import uuid
import warnings
from pathlib import Path

import hdbscan
import numpy as np
from cluster_descriptor.interfaces import ClusterDescriptor
from document_clustering.clustering_metrics import compute_clustering_metrics
from document_clustering.dimensionality_reduction import reduce_with_pca_umap, reduce_with_umap
from document_clustering.document_clustering import Cluster, EmbeddedDocument
from document_clustering.manifold_distortion_analyzer import measure_after_state, measure_before_state

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def cluster_with_pipeline(
    embeddings: np.ndarray,
    docs: list[EmbeddedDocument],
    cluster_descriptor: ClusterDescriptor,
    pipeline_type: str = "umap",
    output_dir: str = "metrics_umap",
    pca_components: int | None = None,
) -> tuple[list[Cluster], np.ndarray]:
    """
    Cluster documents using specified dimensionality reduction pipeline.

    Args:
        embeddings: Original embeddings (N x D)
        docs: List of embedded documents
        cluster_descriptor: Descriptor for generating cluster descriptions
        pipeline_type: "umap" or "pca_umap"
        output_dir: Directory to save metrics
        pca_components: PCA components for PCA+UMAP pipeline

    Returns:
        Tuple of (clusters, labels)
    """
    card = len(embeddings)
    min_cluster_size = 2

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if card < 5:
        single_cluster = Cluster(
            id=str(uuid.uuid4()),
            documents=[doc["document"] for doc in docs],
            description="",
        )
        return [single_cluster], np.array([0] * card)

    if card < 20:
        min_samples = 2
        reduced_embeddings = embeddings
        labels = _perform_clustering(reduced_embeddings, min_cluster_size, min_samples)

        # Save clustering metrics
        clustering_metrics = compute_clustering_metrics(reduced_embeddings, labels)
        clustering_metrics["pipeline_type"] = pipeline_type
        _save_metrics(clustering_metrics, output_dir, "clustering_metrics.json")
    else:
        min_samples = math.floor(math.sqrt(card) / 3)
        umap_n_components = math.floor(min(25, max(2, math.sqrt(embeddings.shape[1]))))

        if embeddings.shape[1] > umap_n_components:
            n_neighbors = min(30, max(5, card // 30))
            k_neighbors = min(15, card // 3)

            # Measure BEFORE reduction
            before_metrics = measure_before_state(embeddings, k_neighbors=k_neighbors, pipeline_type=pipeline_type)
            _save_metrics(before_metrics, output_dir, "distortion_metrics.json")

            # Apply dimensionality reduction
            if pipeline_type == "pca_umap":
                pca_reduced, reduced_embeddings = reduce_with_pca_umap(
                    embeddings,
                    n_components=umap_n_components,
                    n_neighbors=n_neighbors,
                    pca_components=pca_components,
                )

                # Measure AFTER reduction
                after_metrics = measure_after_state(
                    embeddings,
                    reduced_embeddings,
                    k_neighbors=k_neighbors,
                    pipeline_type=pipeline_type,
                    intermediate_dims=pca_reduced.shape[1],
                )
            else:
                reduced_embeddings = reduce_with_umap(
                    embeddings,
                    n_components=umap_n_components,
                    n_neighbors=n_neighbors,
                )

                # Measure AFTER reduction
                after_metrics = measure_after_state(
                    embeddings,
                    reduced_embeddings,
                    k_neighbors=k_neighbors,
                    pipeline_type=pipeline_type,
                )

            _save_metrics(after_metrics, output_dir, "distortion_metrics.json")

            labels = _perform_clustering(reduced_embeddings, min_cluster_size, min_samples)
            clustering_metrics = compute_clustering_metrics(reduced_embeddings, labels)
            clustering_metrics["pipeline_type"] = pipeline_type
            _save_metrics(clustering_metrics, output_dir, "clustering_metrics.json")
        else:
            reduced_embeddings = embeddings
            labels = _perform_clustering(reduced_embeddings, min_cluster_size, min_samples)

            # Save clustering metrics even when no dimensionality reduction
            clustering_metrics = compute_clustering_metrics(reduced_embeddings, labels)
            clustering_metrics["pipeline_type"] = pipeline_type
            _save_metrics(clustering_metrics, output_dir, "clustering_metrics.json")

    clusters = _build_clusters(docs, labels, cluster_descriptor)

    return clusters, labels


def _perform_clustering(embeddings: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        return clusterer.fit_predict(embeddings)


def _build_clusters(
    docs: list[EmbeddedDocument],
    labels: np.ndarray,
    cluster_descriptor: ClusterDescriptor,
) -> list[Cluster]:
    unique_labels = set(labels)
    clusters: dict[str, Cluster] = {}

    for _ in unique_labels:
        cluster_id = str(uuid.uuid4())
        clusters[cluster_id] = Cluster(id=cluster_id, documents=[], description="")

    label_to_cluster_id = {label: cluster_id for label, cluster_id in zip(unique_labels, clusters.keys(), strict=False)}

    for doc, label in zip(docs, labels, strict=False):
        cluster_id = label_to_cluster_id[label]
        clusters[cluster_id]["documents"].append(doc["document"])

    description = cluster_descriptor.describe({cid: cluster["documents"] for cid, cluster in clusters.items()})

    for cid, desc in description.items():
        if cid in clusters:
            clusters[cid]["description"] = desc

    return list(clusters.values())


def _save_metrics(metrics: dict, output_dir: str, filename: str) -> None:
    filepath = Path(output_dir) / filename

    if filepath.exists():
        with open(filepath, "r") as f:
            existing = json.load(f)

        if isinstance(existing, list):
            existing.append(metrics)
        else:
            existing = [existing, metrics]

        data = existing
    else:
        data = [metrics]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compare_pipelines(
    docs: list[EmbeddedDocument],
    cluster_descriptor: ClusterDescriptor,
    base_output_dir: str = "pipeline_comparison",
    pca_components: int | None = None,
) -> dict:
    """
    Compare UMAP vs PCA+UMAP clustering pipelines.

    Args:
        docs: List of embedded documents
        cluster_descriptor: Descriptor for generating cluster descriptions
        base_output_dir: Base directory for outputs
        pca_components: PCA components for PCA+UMAP pipeline

    Returns:
        Dictionary with results from both pipelines
    """
    embeddings = np.array([doc["embedding"] for doc in docs])

    print("Running UMAP-only pipeline...")
    umap_clusters, umap_labels = cluster_with_pipeline(
        embeddings,
        docs,
        cluster_descriptor,
        pipeline_type="umap",
        output_dir=f"{base_output_dir}/umap_pipeline",
    )

    print("Running PCA+UMAP pipeline...")
    pca_umap_clusters, pca_umap_labels = cluster_with_pipeline(
        embeddings,
        docs,
        cluster_descriptor,
        pipeline_type="pca_umap",
        output_dir=f"{base_output_dir}/pca_umap_pipeline",
        pca_components=pca_components,
    )

    return {
        "umap": {
            "clusters": umap_clusters,
            "labels": umap_labels,
            "output_dir": f"{base_output_dir}/umap_pipeline",
        },
        "pca_umap": {
            "clusters": pca_umap_clusters,
            "labels": pca_umap_labels,
            "output_dir": f"{base_output_dir}/pca_umap_pipeline",
        },
    }
