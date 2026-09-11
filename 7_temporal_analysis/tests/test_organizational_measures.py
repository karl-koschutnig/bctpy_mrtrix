"""Tests for organizational_measures.py."""

import numpy as np
import pytest

from scripts.organizational_measures import (
    compute_organizational_measures,
    weights_to_lengths,
)


def _random_connected(n=40, density=0.2, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.random((n, n))
    W[rng.random((n, n)) > density] = 0
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    # guarantee connectivity: add a ring
    for i in range(n):
        W[i, (i + 1) % n] = W[(i + 1) % n, i] = 0.5
    return W


def test_weights_to_lengths_preserves_zeros_and_inverts_edges():
    W = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 0.5], [0.0, 0.5, 0.0]])
    L = weights_to_lengths(W)
    assert L[0, 1] == pytest.approx(0.5)   # 1 / 2.0
    assert L[1, 2] == pytest.approx(2.0)   # 1 / 0.5
    assert L[0, 2] == 0.0                  # no edge stays 0, not 1/eps


def test_all_measures_present_and_finite_on_connected_graph():
    W = _random_connected()
    m = compute_organizational_measures(W)
    expected = {
        "density", "path_length", "global_efficiency", "modularity",
        "core_periphery", "kcore_size", "score_size", "strength_mean",
        "local_efficiency_mean", "clustering_coef_mean", "betweenness_mean",
        "subgraph_centrality_log10_mean",
    }
    assert set(m) == expected
    for k, v in m.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"
    # sane ranges
    assert 0 < m["density"] < 1
    assert 0 < m["global_efficiency"] <= 1
    assert 0 <= m["modularity"] <= 1
    assert 0 <= m["clustering_coef_mean"] <= 1
    assert m["path_length"] > 1


def test_disconnected_graph_nans_path_length_but_keeps_local_measures():
    W = _random_connected()
    W[5, :] = 0
    W[:, 5] = 0  # node 5 isolated -> whole graph disconnected
    with pytest.warns(UserWarning, match="Disconnected graph"):
        m = compute_organizational_measures(W)
    assert np.isnan(m["path_length"])
    assert np.isnan(m["global_efficiency"])
    # measures that don't depend on global reachability still compute
    assert np.isfinite(m["clustering_coef_mean"])
    assert np.isfinite(m["modularity"])
    assert np.isfinite(m["density"])


def test_empty_graph_returns_all_nan():
    m = compute_organizational_measures(np.zeros((10, 10)))
    assert all(np.isnan(v) for v in m.values())
