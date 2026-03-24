"""Correlation matrix construction: sample, Cfull, C0, and regularized."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_correlation_matrix(z: np.ndarray) -> np.ndarray:
    """Compute sample correlation matrix from standardized returns.

    Args:
        z: (T, N) standardized return matrix

    Returns:
        (N, N) correlation matrix
    """
    T, N = z.shape
    C = (z.T @ z) / T
    # Force symmetry
    C = (C + C.T) / 2
    # Force diagonal = 1
    d = np.sqrt(np.diag(C))
    d = np.where(d < 1e-12, 1.0, d)
    D_inv = np.diag(1.0 / d)
    C = D_inv @ C @ D_inv
    np.fill_diagonal(C, 1.0)
    return C


def compute_cfull(
    cc_returns: pd.DataFrame,
    start_date: str,
    end_date: str,
    window: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute long-term correlation matrix Cfull over a fixed period.

    Uses pairwise correlation to handle tickers with different data availability
    (e.g., XLC starts 2018, XLRE starts 2015).

    For tickers without data in the Cfull period, uses identity-like entries
    (zero off-diagonal correlation) as a neutral prior.

    Args:
        cc_returns: Close-to-close returns (date x all tickers)
        start_date: Start of Cfull estimation period
        end_date: End of Cfull estimation period
        window: Not used for Cfull (full period standardization)
        eps: Floor for std

    Returns:
        (N, N) correlation matrix
    """
    subset = cc_returns.loc[start_date:end_date].dropna(how="all")
    N = subset.shape[1]

    # Standardize each column over its available data
    mu = subset.mean()
    sigma = subset.std(ddof=0).clip(lower=eps)
    z = (subset - mu) / sigma

    # Pairwise correlation: for each pair, use overlapping non-NaN dates
    C = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            mask = z.iloc[:, i].notna() & z.iloc[:, j].notna()
            if mask.sum() < 10:
                # Insufficient overlap: use zero correlation
                C[i, j] = 0.0
                C[j, i] = 0.0
            else:
                zi = z.iloc[:, i][mask].values
                zj = z.iloc[:, j][mask].values
                corr = np.dot(zi, zj) / len(zi)
                C[i, j] = corr
                C[j, i] = corr

    # Ensure positive semi-definite by clamping negative eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    C = (C + C.T) / 2
    # Re-normalize to correlation
    d = np.sqrt(np.diag(C))
    d = np.where(d < 1e-12, 1.0, d)
    D_inv = np.diag(1.0 / d)
    C = D_inv @ C @ D_inv
    np.fill_diagonal(C, 1.0)
    return C


def build_C0(V0: np.ndarray, Cfull: np.ndarray) -> np.ndarray:
    """Build prior exposure matrix C0.

    C0_raw = V0 @ D0 @ V0^T
    where D0 = diag(V0^T @ Cfull @ V0)
    Then normalize to correlation matrix (diag = 1).

    Args:
        V0: (N, K0) prior eigenvector matrix
        Cfull: (N, N) long-term correlation matrix

    Returns:
        (N, N) prior correlation matrix C0
    """
    # D0: diagonal of V0^T Cfull V0
    proj = V0.T @ Cfull @ V0  # (K0, K0)
    D0 = np.diag(np.diag(proj))  # Keep only diagonal

    # C0_raw = V0 D0 V0^T
    C0_raw = V0 @ D0 @ V0.T

    # Normalize to correlation matrix
    delta = np.diag(C0_raw).copy()
    delta = np.where(delta < 1e-12, 1e-12, delta)
    delta_inv_sqrt = np.diag(1.0 / np.sqrt(delta))
    C0 = delta_inv_sqrt @ C0_raw @ delta_inv_sqrt

    # Force symmetry and diagonal = 1
    C0 = (C0 + C0.T) / 2
    np.fill_diagonal(C0, 1.0)

    return C0


def regularize_correlation(
    Ct: np.ndarray,
    C0: np.ndarray,
    lambda_reg: float,
) -> np.ndarray:
    """Compute regularized correlation matrix.

    C_reg = (1 - lambda) * Ct + lambda * C0

    Args:
        Ct: (N, N) sample correlation matrix at time t
        C0: (N, N) prior correlation matrix
        lambda_reg: Regularization parameter in [0, 1]

    Returns:
        (N, N) regularized correlation matrix
    """
    C_reg = (1 - lambda_reg) * Ct + lambda_reg * C0
    C_reg = (C_reg + C_reg.T) / 2
    np.fill_diagonal(C_reg, 1.0)
    return C_reg
