"""Tests for PCA_SUB model."""

import numpy as np
import pytest

from src.covariance import compute_correlation_matrix
from src.pca_plain import eigen_decompose, split_us_jp_loadings
from src.pca_sub import PCASubModel
from src.constants import N_US, N_JP


def test_eigen_descending():
    """Eigenvalues should be in descending order."""
    rng = np.random.default_rng(42)
    z = rng.standard_normal((200, 28))
    C = compute_correlation_matrix(z)
    vals, _ = eigen_decompose(C, 5)
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_split_dimensions():
    """Block split should give correct dimensions."""
    rng = np.random.default_rng(42)
    V_K = rng.standard_normal((28, 3))
    V_U, V_J = split_us_jp_loadings(V_K, N_US, N_JP)
    assert V_U.shape == (N_US, 3)
    assert V_J.shape == (N_JP, 3)


def test_pca_sub_signal_shape():
    """PCA_SUB signal should have N_JP elements."""
    rng = np.random.default_rng(42)
    Cfull = compute_correlation_matrix(rng.standard_normal((500, 28)))

    model = PCASubModel(Cfull=Cfull, n_us=N_US, n_jp=N_JP)

    Ct = compute_correlation_matrix(rng.standard_normal((60, 28)))
    z_us = rng.standard_normal(N_US)
    signal = model.compute_signal(Ct, z_us)
    assert signal.shape == (N_JP,)


def test_lambda_zero_approx_plain():
    """With lambda=0, PCA_SUB should produce similar results to PCA_PLAIN."""
    rng = np.random.default_rng(42)
    Cfull = compute_correlation_matrix(rng.standard_normal((500, 28)))

    model = PCASubModel(Cfull=Cfull, n_us=N_US, n_jp=N_JP, lambda_reg=0.0)

    Ct = compute_correlation_matrix(rng.standard_normal((60, 28)))
    z_us = rng.standard_normal(N_US)

    sig_sub = model.compute_signal(Ct, z_us)

    # PCA_PLAIN directly
    vals, V_K = eigen_decompose(Ct, 3)
    V_U, V_J = split_us_jp_loadings(V_K, N_US, N_JP)
    from src.pca_plain import compute_pca_signal
    sig_plain = compute_pca_signal(z_us, V_U, V_J)

    # Should be very close (may differ due to sign convention)
    # Check if they're close in absolute value or with sign flip
    corr = np.abs(np.corrcoef(sig_sub, sig_plain)[0, 1])
    assert corr > 0.99, f"Correlation between lambda=0 PCA_SUB and PCA_PLAIN: {corr}"
