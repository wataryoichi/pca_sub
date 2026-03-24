"""PCA_SUB: Subspace-regularized PCA strategy (the main method)."""

from __future__ import annotations

import numpy as np

from .covariance import build_C0, compute_correlation_matrix, regularize_correlation
from .pca_plain import compute_pca_signal, eigen_decompose, split_us_jp_loadings
from .prior_factors import build_V0


class PCASubModel:
    """Stateful model for subspace-regularized PCA signal generation.

    Pre-computes C0 from Cfull and V0, then for each time step:
    1. Regularize the sample correlation matrix: C_reg = (1-λ)Ct + λC0
    2. Eigendecompose C_reg, take top K
    3. Split into US/JP blocks
    4. Project US standardized returns -> JP signal
    """

    def __init__(
        self,
        Cfull: np.ndarray,
        tickers: list[str] | None = None,
        n_us: int = 11,
        n_jp: int = 17,
        lambda_reg: float = 0.9,
        n_components: int = 3,
    ) -> None:
        self.n_us = n_us
        self.n_jp = n_jp
        self.lambda_reg = lambda_reg
        self.n_components = n_components

        # Build prior subspace and C0
        self.V0 = build_V0(tickers=tickers, n_us=n_us, n_jp=n_jp)
        self.C0 = build_C0(self.V0, Cfull)

    def compute_signal(
        self,
        Ct: np.ndarray,
        z_us: np.ndarray,
    ) -> np.ndarray:
        """Compute PCA_SUB signal for one time step.

        Args:
            Ct: (N, N) sample correlation matrix at time t
            z_us: (N_US,) standardized US returns at time t

        Returns:
            (N_JP,) predicted JP signal
        """
        C_reg = regularize_correlation(Ct, self.C0, self.lambda_reg)
        _, V_K = eigen_decompose(C_reg, self.n_components)
        V_U, V_J = split_us_jp_loadings(V_K, self.n_us, self.n_jp)
        return compute_pca_signal(z_us, V_U, V_J)
