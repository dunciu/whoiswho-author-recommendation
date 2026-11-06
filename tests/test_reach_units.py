"""Unit tests for the data handling and scoring helpers in reach.py."""
import math
from collections import Counter

import numpy as np
import pytest
import torch

import reach


def test_parse_authors_dedupes_and_prefers_ids():
    out = reach.parse_authors([
        {"name": "A", "org": "X"},
        {"id": "7", "name": "B"},
        "C",
        {"name": "A", "org": "Y"},   # duplicate author
        {"org": "no id or name"},    # skipped
    ])
    assert out == [("A", "X"), ("7", ""), ("C", "")]


def test_parse_authors_handles_bad_input():
    assert reach.parse_authors(None) == []
    assert reach.parse_authors([1, None]) == []


@pytest.mark.parametrize("val,expected", [(2015, 2015.0), ("2018", 2018.0), ("", 2020.0), (None, 2020.0), ("n/a", 2020.0)])
def test_safe_year(val, expected):
    assert reach.safe_year(val) == expected


def test_kcore_prune_satisfies_core_property():
    papers = reach.generate_synthetic(1500, seed=0)
    pruned = reach.kcore_prune(papers, k=3)
    assert pruned, "pruning removed everything"
    deg = Counter(a for p in pruned for a in p["clean"])
    assert min(deg.values()) >= 3
    assert min(len(p["clean"]) for p in pruned) >= 3
    for p in pruned:
        assert set(p["org_of"]) == set(p["clean"])


def test_pca_reduce_shape_norm_and_padding():
    X = np.random.default_rng(0).normal(size=(200, 16)).astype(np.float32)
    Z = reach.pca_reduce(X, 32, seed=0)
    assert Z.shape == (200, 32)
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-4)
    assert np.all(Z[:, 16:] == 0)          # only 16 components exist
    assert reach.pca_reduce(np.zeros((0, 8), np.float32), 4, 0).shape == (0, 4)


def test_ndcg_recall_prefix_matches_hand_calculation():
    res = reach.ndcg_recall_prefix([3, 5, 1, 9], {5, 9}, [1, 2, 4])
    assert res[1] == (0.0, 0.0)
    dcg2, idcg2 = 1 / math.log2(3), 1 + 1 / math.log2(3)
    assert res[2][0] == pytest.approx(dcg2 / idcg2)
    assert res[2][1] == 0.5
    assert res[4][1] == 1.0


def test_minmax_ignores_inf():
    s = torch.tensor([1.0, 3.0, -float("inf"), 2.0])
    out = reach.minmax(s)
    assert torch.allclose(out, torch.tensor([0.0, 1.0, 0.0, 0.5]))
    assert torch.all(reach.minmax(torch.full((3,), -float("inf"))) == 0)


def test_fuse_drops_rule_expert_at_cold_start_and_exclusions():
    stack = torch.tensor([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])   # rows: GNN, SEM, RB
    w = torch.tensor([0.5, 0.5, 1.0])
    cold = reach.fuse(stack, w, ctx=0)
    assert torch.allclose(cold, torch.tensor([0.5, 0.5]))          # RB weight forced to 0
    no_sem = reach.fuse(stack, w, ctx=1, exclude=("S",))
    assert torch.allclose(no_sem, (0.5 * stack[0] + 1.0 * stack[2]) / 1.5)


def test_decayed_lightgcn_forward_and_lambda_gradient():
    torch.manual_seed(0)
    n_p, n_a, d_in = 6, 4, 8
    a_idx = torch.tensor([0, 1, 1, 2, 3, 3, 0])
    p_idx = torch.tensor([0, 0, 1, 2, 3, 4, 5])
    age = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    model = reach.DecayedLightGCN(d_in, d_in, hidden=16, learn_decay=True)
    hp, ha, hp0, ha0 = model(torch.randn(n_p, d_in), torch.randn(n_a, d_in), a_idx, p_idx, age)
    assert hp.shape == (n_p, 16) and ha.shape == (n_a, 16)
    assert torch.allclose(hp.norm(dim=1), torch.ones(n_p), atol=1e-5)
    assert model.decay_rate().item() == pytest.approx(reach.RB_DECAY, rel=1e-4)
    (hp.sum() + ha.sum()).backward()
    assert model.raw_lambda.grad is not None


def test_decay_disabled_gives_binary_weights():
    model = reach.DecayedLightGCN(4, 4, hidden=8, learn_decay=False)
    assert model.decay_rate().item() == 0.0
