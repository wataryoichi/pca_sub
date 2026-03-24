"""PCA_PLAIN strategy: standard PCA without regularization (lambda=0)."""

from __future__ import annotations

import numpy as np

from .constants import N_JP, N_US


def eigen_decompose(C: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a symmetric matrix, return top-K eigenvectors/values.

    Args:
        C: (N, N) symmetric matrix
        n_components: K, number of components to keep

    Returns:
        eigenvalues: (K,) in descending order
        eigenvectors: (N, K) columns are eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # eigh returns ascending order; reverse to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx[:n_components]]
    eigenvectors = eigenvectors[:, idx[:n_components]]
    # Ensure real (should be for symmetric)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    return eigenvalues, eigenvectors


def split_us_jp_loadings(
    V_K: np.ndarray,
    n_us: int = N_US,
    n_jp: int = N_JP,
) -> tuple[np.ndarray, np.ndarray]:
    """Split eigenvector matrix into US and JP blocks.

    V_K = [V_U; V_J]

    Args:
        V_K: (N, K) eigenvector matrix
        n_us: Number of US sectors
        n_jp: Number of JP sectors

    Returns:
        V_U: (N_US, K)
        V_J: (N_JP, K)
    """
    V_U = V_K[:n_us, :]
    V_J = V_K[n_us : n_us + n_jp, :]
    return V_U, V_J


def compute_pca_signal(
    z_us: np.ndarray,
    V_U: np.ndarray,
    V_J: np.ndarray,
) -> np.ndarray:
    """Compute PCA-based lead-lag signal.

    f_t = V_U^T @ z_U,t          (factor scores)
    z_hat_J = V_J @ f_t           (JP signal)

    Args:
        z_us: (N_US,) standardized US returns at time t
        V_U: (N_US, K) US loadings
        V_J: (N_JP, K) JP loadings

    Returns:
        (N_JP,) predicted JP signal
    """
    f = V_U.T @ z_us   # (K,)
    return V_J @ f      # (N_JP,)
