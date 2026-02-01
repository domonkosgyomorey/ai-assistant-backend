import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict:
    valid_mask = labels != -1
    valid_embeddings = embeddings[valid_mask]
    valid_labels = labels[valid_mask]

    n_clusters = len(set(valid_labels))
    n_noise = np.sum(labels == -1)

    metrics = {
        "n_clusters": int(n_clusters),
        "n_noise_points": int(n_noise),
        "noise_ratio": float(n_noise / len(labels)),
    }

    metrics["silhouette_score"] = float(silhouette_score(valid_embeddings, valid_labels))
    metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(valid_embeddings, valid_labels))
    metrics["davies_bouldin_score"] = float(davies_bouldin_score(valid_embeddings, valid_labels))

    return metrics
