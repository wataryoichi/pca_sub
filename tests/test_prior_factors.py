"""Tests for prior factor construction."""

import numpy as np
import pytest

from src.prior_factors import build_V0, build_v1, build_v2, build_v3
from src.constants import N_US, N_JP, N_TOTAL


def test_v1_normalized():
    v = build_v1()
    assert abs(np.linalg.norm(v) - 1.0) < 1e-10


def test_v1_equal_weights():
    v = build_v1()
    assert np.allclose(v, v[0])  # All elements equal


def test_v2_orthogonal_to_v1():
    v1 = build_v1()
    v2 = build_v2()
    assert abs(np.dot(v1, v2)) < 1e-10


def test_v2_normalized():
    v2 = build_v2()
    assert abs(np.linalg.norm(v2) - 1.0) < 1e-10


def test_v3_orthogonal_to_v1_v2():
    v1 = build_v1()
    v2 = build_v2()
    v3 = build_v3()
    assert abs(np.dot(v1, v3)) < 1e-10
    assert abs(np.dot(v2, v3)) < 1e-10


def test_v3_normalized():
    v3 = build_v3()
    assert abs(np.linalg.norm(v3) - 1.0) < 1e-10


def test_V0_orthonormal():
    V0 = build_V0()
    assert V0.shape == (N_TOTAL, 3)
    gram = V0.T @ V0
    assert np.allclose(gram, np.eye(3), atol=1e-10)


def test_V0_dimensions():
    V0 = build_V0()
    assert V0.shape[0] == N_US + N_JP
    assert V0.shape[1] == 3
