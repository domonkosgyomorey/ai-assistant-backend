"""
Manifold distortion analyzer for measuring UMAP's impact on embedding geometry.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances


def measure_before_state(embeddings: np.ndarray, k_neighbors: int = 15, pipeline_type: str = "umap") -> dict:
    """
    Measure embedding manifold state BEFORE dimensionality reduction.

    Args:
        embeddings: Original high-dimensional embeddings (N x D)
        k_neighbors: Number of neighbors for local metrics
        pipeline_type: Type of pipeline ("umap" or "pca_umap")

    Returns:
        Dictionary with metrics
    """
    cosine_dist = pairwise_distances(embeddings, metric="cosine")

    # Local density (k-NN average distance)
    local_densities = []
    for i in range(len(embeddings)):
        knn_dists = np.sort(cosine_dist[i])[1 : k_neighbors + 1]
        local_densities.append(float(knn_dists.mean()))

    return {
        "stage": "before_reduction",
        "pipeline_type": pipeline_type,
        "n_samples": int(len(embeddings)),
        "n_dimensions": int(embeddings.shape[1]),
        "k_neighbors": int(k_neighbors),
        "mean_local_density": float(np.mean(local_densities)),
        "std_local_density": float(np.std(local_densities)),
        "min_local_density": float(np.min(local_densities)),
        "max_local_density": float(np.max(local_densities)),
    }


def measure_after_state(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    k_neighbors: int = 15,
    pipeline_type: str = "umap",
    intermediate_dims: int | None = None,
) -> dict:
    """
    Measure embedding manifold state AFTER dimensionality reduction and compute distortion metrics.

    Args:
        embeddings_before: Original high-dimensional embeddings (N x D)
        embeddings_after: Reduced embeddings (N x d)
        k_neighbors: Number of neighbors for local metrics
        pipeline_type: Type of pipeline ("umap" or "pca_umap")
        intermediate_dims: Intermediate dimensions (e.g., PCA dims in PCA+UMAP)

    Returns:
        Dictionary with distortion metrics
    """
    cosine_dist = pairwise_distances(embeddings_before, metric="cosine")
    l2_dist = pairwise_distances(embeddings_after, metric="euclidean")

    # 1. Trustworthiness
    trust = trustworthiness(embeddings_before, embeddings_after, n_neighbors=k_neighbors)

    # 2. Geodesic preservation (k-NN distance correlation)
    knn_indices = np.argsort(cosine_dist, axis=1)[:, 1 : k_neighbors + 1]
    correlations = []
    for i in range(len(embeddings_before)):
        knn_idx = knn_indices[i]
        if len(knn_idx) > 1:
            corr = spearmanr(cosine_dist[i, knn_idx], l2_dist[i, knn_idx]).correlation
            if not np.isnan(corr):
                correlations.append(corr)

    geo_preservation = float(np.mean(correlations)) if correlations else 0.0

    # 3. Local density preservation
    def compute_densities(X, metric):
        dist = pairwise_distances(X, metric=metric)
        densities = []
        for i in range(len(X)):
            knn_dists = np.sort(dist[i])[1 : k_neighbors + 1]
            densities.append(knn_dists.mean())
        return np.array(densities)

    orig_density = compute_densities(embeddings_before, "cosine")
    reduced_density = compute_densities(embeddings_after, "euclidean")

    # Normalize for correlation
    orig_norm = (orig_density - orig_density.min()) / (orig_density.max() - orig_density.min() + 1e-10)
    reduced_norm = (reduced_density - reduced_density.min()) / (reduced_density.max() - reduced_density.min() + 1e-10)

    density_preservation = float(np.corrcoef(orig_norm, reduced_norm)[0, 1])

    # 4. Distance ratio statistics
    distance_ratios = []
    for i in range(len(embeddings_before)):
        knn_idx = knn_indices[i]
        for j in knn_idx:
            orig_dist = cosine_dist[i, j]
            reduced_dist = l2_dist[i, j]
            if orig_dist > 1e-10:
                distance_ratios.append(reduced_dist / orig_dist)

    mean_ratio = float(np.mean(distance_ratios)) if distance_ratios else 1.0
    std_ratio = float(np.std(distance_ratios)) if distance_ratios else 0.0

    # 5. k-NN preservation
    knn_after_indices = np.argsort(l2_dist, axis=1)[:, 1 : k_neighbors + 1]
    preservation_rates = []
    for i in range(len(embeddings_before)):
        before_knn = set(knn_indices[i])
        after_knn = set(knn_after_indices[i])
        intersection = len(before_knn & after_knn)
        preservation_rates.append(intersection / k_neighbors)

    knn_preservation = float(np.mean(preservation_rates))

    result = {
        "stage": "after_reduction",
        "pipeline_type": pipeline_type,
        "n_samples": int(len(embeddings_after)),
        "n_dimensions": int(embeddings_after.shape[1]),
        "k_neighbors": int(k_neighbors),
        "trustworthiness": float(trust),
        "geodesic_preservation": float(geo_preservation),
        "density_preservation": float(density_preservation),
        "mean_distance_ratio": float(mean_ratio),
        "std_distance_ratio": float(std_ratio),
        "knn_preservation": float(knn_preservation),
        "mean_local_density": float(reduced_density.mean()),
        "std_local_density": float(reduced_density.std()),
    }

    if intermediate_dims is not None:
        result["intermediate_dims"] = int(intermediate_dims)

    return result


def save_metrics(metrics: dict, filepath: str, append: bool = True) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        filepath: Path to JSON file
        append: If True, append to existing file; if False, overwrite
    """
    path = Path(filepath)

    if append and path.exists():
        # Load existing metrics
        with open(path, "r") as f:
            existing = json.load(f)

        # Append new metrics
        if isinstance(existing, list):
            existing.append(metrics)
        else:
            existing = [existing, metrics]

        data = existing
    else:
        # Create new file with single metric
        data = [metrics]

    # Save
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
