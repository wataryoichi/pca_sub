"""Tests for covariance matrix construction."""

import numpy as np
import pytest

from src.covariance import build_C0, compute_correlation_matrix, regularize_correlation
from src.prior_factors import build_V0


def test_correlation_matrix_symmetric():
    rng = np.random.default_rng(42)
    z = rng.standard_normal((100, 28))
    C = compute_correlation_matrix(z)
    assert np.allclose(C, C.T, atol=1e-10)


def test_correlation_matrix_diagonal_ones():
    rng = np.random.default_rng(42)
    z = rng.standard_normal((100, 28))
    C = compute_correlation_matrix(z)
    assert np.allclose(np.diag(C), 1.0, atol=1e-10)


def test_C0_symmetric():
    rng = np.random.default_rng(42)
    V0 = build_V0()
    Cfull = compute_correlation_matrix(rng.standard_normal((500, 28)))
    C0 = build_C0(V0, Cfull)
    assert np.allclose(C0, C0.T, atol=1e-10)


def test_C0_diagonal_ones():
    rng = np.random.default_rng(42)
    V0 = build_V0()
    Cfull = compute_correlation_matrix(rng.standard_normal((500, 28)))
    C0 = build_C0(V0, Cfull)
    assert np.allclose(np.diag(C0), 1.0, atol=1e-10)


def test_regularize_lambda_zero_returns_Ct():
    rng = np.random.default_rng(42)
    Ct = compute_correlation_matrix(rng.standard_normal((100, 28)))
    C0 = np.eye(28)
    C_reg = regularize_correlation(Ct, C0, lambda_reg=0.0)
    assert np.allclose(C_reg, Ct, atol=1e-10)


def test_regularize_lambda_one_returns_C0():
    rng = np.random.default_rng(42)
    Ct = compute_correlation_matrix(rng.standard_normal((100, 28)))
    C0 = np.eye(28)
    C_reg = regularize_correlation(Ct, C0, lambda_reg=1.0)
    assert np.allclose(C_reg, C0, atol=1e-10)
