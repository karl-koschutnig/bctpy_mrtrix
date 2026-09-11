"""Tests for hub_measures.py - hub-set definition and the scalar measures."""
import numpy as np
import pytest

from scripts.hub_measures import HUB_KEYS, compute_hub_measures


def _hub_periphery_graph(n_hub=4, n_periph=16, seed=0):
    """A graph where the first n_hub nodes are densely, heavily interconnected
    (a rich club) and the periphery is sparse and light."""
    rng = np.random.default_rng(seed)
    n = n_hub + n_periph
    W = np.zeros((n, n))
    for i in range(n_hub):
        for j in range(i + 1, n_hub):
            W[i, j] = W[j, i] = rng.uniform(5, 10)          # heavy hub-hub
    for i in range(n_hub):
        for j in range(n_hub, n):
            if rng.random() < 0.5:
                W[i, j] = W[j, i] = rng.uniform(0.5, 1.5)   # feeder
    for i in range(n_hub, n):
        for j in range(i + 1, n):
            if rng.random() < 0.15:
                W[i, j] = W[j, i] = rng.uniform(0.1, 0.4)   # sparse periphery
    return W


def test_all_keys_present_and_finite_on_hub_graph():
    W = _hub_periphery_graph()
    hub_idx = np.array([0, 1, 2, 3])
    out = compute_hub_measures(W, hub_idx)
    assert set(out) == set(HUB_KEYS)
    for k in ("hub_strength_frac", "richclub_edge_frac", "feeder_edge_frac",
              "local_edge_frac", "hub_participation_mean"):
        assert np.isfinite(out[k]), k


def test_edge_fractions_sum_to_one():
    W = _hub_periphery_graph()
    out = compute_hub_measures(W, np.array([0, 1, 2, 3]))
    total = out["richclub_edge_frac"] + out["feeder_edge_frac"] + out["local_edge_frac"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_hubs_carry_disproportionate_strength():
    W = _hub_periphery_graph()
    out = compute_hub_measures(W, np.array([0, 1, 2, 3]))
    # 4 of 20 nodes (20%) but they are the heavy ones -> well above 20% of strength
    assert out["hub_strength_frac"] > 0.4


def test_empty_graph_returns_all_nan():
    out = compute_hub_measures(np.zeros((10, 10)), np.array([0, 1]))
    assert all(np.isnan(v) for v in out.values())
