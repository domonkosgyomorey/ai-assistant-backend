import warnings

import numpy as np
import umap.umap_ as umap
from sklearn.decomposition import PCA


def reduce_with_umap(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float = 0.05,
) -> np.ndarray:
    """
    Reduce dimensionality using UMAP only.

    Args:
        embeddings: Original embeddings (N x D)
        n_components: Target dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter

    Returns:
        Reduced embeddings (N x n_components)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            output_metric="euclidean",
        )
        return umap_reducer.fit_transform(embeddings)


def reduce_with_pca_umap(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    pca_components: int | None = None,
    min_dist: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce dimensionality using PCA then UMAP.

    Args:
        embeddings: Original embeddings (N x D)
        n_components: Final target dimensions (after UMAP)
        n_neighbors: UMAP n_neighbors parameter
        pca_components: Intermediate PCA dimensions (if None, uses min(100, D//2))
        min_dist: UMAP min_dist parameter

    Returns:
        Tuple of (PCA reduced embeddings, final UMAP reduced embeddings)
    """
    if pca_components is None:
        pca_components = min(100, embeddings.shape[1] // 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = PCA(n_components=pca_components)
        pca_reduced = pca.fit_transform(embeddings)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",  # PCA output is already Euclidean
            output_metric="euclidean",
        )
        umap_reduced = umap_reducer.fit_transform(pca_reduced)

    return pca_reduced, umap_reduced
