"""Prior subspace V0 construction for subspace-regularized PCA.

Three prior eigenvectors:
1. v1: Global factor (equal weight on all sectors)
2. v2: Country spread factor (US positive, JP negative)
3. v3: Cyclical/Defensive factor
"""

from __future__ import annotations

import numpy as np

from .constants import ALL_TICKERS, CYCLICAL_DEFENSIVE, N_JP, N_TOTAL, N_US


def build_v1(n: int = N_TOTAL) -> np.ndarray:
    """Global factor: equal weight, normalized."""
    v = np.ones(n)
    return v / np.linalg.norm(v)


def build_v2(n_us: int = N_US, n_jp: int = N_JP) -> np.ndarray:
    """Country spread factor: US=+1, JP=-1, orthogonalized to v1."""
    n = n_us + n_jp
    v1 = build_v1(n)

    a = np.concatenate([np.ones(n_us), -np.ones(n_jp)])
    # Gram-Schmidt: remove v1 component
    a_orth = a - np.dot(v1, a) * v1
    norm = np.linalg.norm(a_orth)
    if norm < 1e-12:
        raise ValueError("v2 is degenerate after orthogonalization")
    return a_orth / norm


def build_v3(
    tickers: list[str] | None = None,
    n_us: int = N_US,
    n_jp: int = N_JP,
) -> np.ndarray:
    """Cyclical/Defensive factor, orthogonalized to v1 and v2.

    Uses the classification from constants.CYCLICAL_DEFENSIVE.
    """
    if tickers is None:
        tickers = ALL_TICKERS

    n = n_us + n_jp
    v1 = build_v1(n)
    v2 = build_v2(n_us, n_jp)

    b = np.array([CYCLICAL_DEFENSIVE.get(t, 0) for t in tickers], dtype=float)

    # Gram-Schmidt: remove v1 and v2 components
    b_orth = b - np.dot(v1, b) * v1 - np.dot(v2, b) * v2
    norm = np.linalg.norm(b_orth)
    if norm < 1e-12:
        raise ValueError("v3 is degenerate after orthogonalization")
    return b_orth / norm


def build_V0(
    tickers: list[str] | None = None,
    n_us: int = N_US,
    n_jp: int = N_JP,
) -> np.ndarray:
    """Build prior subspace V0 = [v1, v2, v3] as (N, K0) matrix.

    Each column is a normalized, mutually orthogonal prior eigenvector.
    """
    if tickers is None:
        tickers = ALL_TICKERS

    v1 = build_v1(n_us + n_jp)
    v2 = build_v2(n_us, n_jp)
    v3 = build_v3(tickers, n_us, n_jp)

    V0 = np.column_stack([v1, v2, v3])

    # Verify orthonormality
    gram = V0.T @ V0
    assert np.allclose(gram, np.eye(3), atol=1e-10), (
        f"V0 is not orthonormal: V0'V0 =\n{gram}"
    )

    return V0
