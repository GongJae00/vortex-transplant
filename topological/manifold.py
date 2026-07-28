"""Manifold diagnostics.

Tests whether intervened hidden states lie on the natural manifold
of hidden states produced by the trained model.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from .types import ManifoldModel, ManifoldDiagnostics


def fit_manifold_model(
    natural_states: list[np.ndarray],  # list of (2*C*H*W,) flattened real states
    method: str = "pca",
    n_components: int | None = None,
    knn_k: int = 5,
) -> ManifoldModel:
    """Fit a manifold model to natural hidden states."""
    X = np.stack(natural_states)  # (N, D)
    N, D = X.shape

    if n_components is None:
        n_components = min(D, int(0.95 * N))

    pca = PCA(n_components=n_components)
    pca.fit(X)

    return ManifoldModel(
        method=method,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_ratio_,
        natural_pool=natural_states,
        knn_k=knn_k,
    )


def manifold_diagnostics(
    intervened_state: np.ndarray,   # (2*C*H*W,) flattened real
    model: ManifoldModel,
    natural_pool: list[np.ndarray] | None = None,
    natural_neighbor_distance_threshold: float | None = None,
) -> ManifoldDiagnostics:
    """Compute manifold validity diagnostics for an intervened state."""
    pool = natural_pool if natural_pool is not None else model.natural_pool
    if pool is None or len(pool) == 0:
        return ManifoldDiagnostics(
            reconstruction_error=0.0,
            knn_density_ratio=1.0,
            nearest_natural_distance=0.0,
            relaxation_drift=0.0,
            on_manifold=True,  # cannot determine without pool
        )

    X_pool = np.stack(pool)

    # PCA reconstruction error
    if model.pca_components is not None and model.pca_mean is not None:
        centered = intervened_state - model.pca_mean
        projected = centered @ model.pca_components.T @ model.pca_components
        reconstruction_error = float(np.linalg.norm(centered - projected))
    else:
        reconstruction_error = 0.0

    # kNN density ratio
    knn = NearestNeighbors(n_neighbors=model.knn_k)
    knn.fit(X_pool)

    # Average distance to k nearest natural states
    dist_to_natural, _ = knn.kneighbors(intervened_state.reshape(1, -1))
    nearest_natural_distance = float(np.mean(dist_to_natural))

    # Natural-natural reference distances (sample-based)
    sample_idx = np.random.choice(len(pool), size=min(100, len(pool)), replace=False)
    X_sample = X_pool[sample_idx]
    nat_distances, _ = knn.kneighbors(X_sample)
    nat_mean_dist = float(np.mean(nat_distances))

    # kNN density ratio: natural / intervened
    if nearest_natural_distance > 0:
        knn_density_ratio = nat_mean_dist / nearest_natural_distance
    else:
        knn_density_ratio = float("inf")

    # On-manifold decision
    if natural_neighbor_distance_threshold is not None:
        on_manifold = nearest_natural_distance <= natural_neighbor_distance_threshold
    else:
        # Default: within 2x of natural spread
        on_manifold = nearest_natural_distance <= 2.0 * nat_mean_dist

    return ManifoldDiagnostics(
        reconstruction_error=reconstruction_error,
        knn_density_ratio=float(knn_density_ratio),
        nearest_natural_distance=nearest_natural_distance,
        relaxation_drift=0.0,  # requires model access, deferred to evaluation
        on_manifold=on_manifold,
    )


def compute_natural_neighbor_threshold(
    natural_pool: list[np.ndarray],
    percentile: float = 95.0,
    knn_k: int = 5,
) -> float:
    """Compute distance threshold for on-manifold classification.

    Returns the p-th percentile of natural-natural neighbor distances.
    States with nearest natural distance ≤ this threshold are considered
    on-manifold.
    """
    X = np.stack(natural_pool)
    knn = NearestNeighbors(n_neighbors=knn_k + 1)  # +1 to skip self
    knn.fit(X)
    distances, _ = knn.kneighbors(X)
    # Remove self-distance (first column)
    neighbor_dists = distances[:, 1:].ravel()
    return float(np.percentile(neighbor_dists, percentile))
