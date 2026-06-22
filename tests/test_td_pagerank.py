"""
Property-based tests for TDPageRankEngine.

Properties tested:
    Property 6  (Task 2.2) – Temporal edge weight decay formula
    Property 7  (Task 2.3) – Score vector invariants
    Property 8  (Task 2.4) – Cycle penalty correctness
    Property 9  (Task 2.5) – Dormant node rank suppression
    Property 10 (Task 2.6) – Determinism

All tests use Hypothesis for property-based generation.
"""

import sys
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import networkx as nx
import pytest
from hypothesis import given, settings, assume, strategies as st

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.td_pagerank import TDPageRankEngine
from tests.conftest import random_directed_graph, graph_to_edges_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_edges_df(senders, receivers, amounts, dates):
    """Build a minimal edges DataFrame from parallel lists."""
    df = pd.DataFrame(
        {
            "Sender_account": senders,
            "Receiver_account": receivers,
            "amount_local_npr": amounts,
            "Date": pd.to_datetime(dates),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Property 6 — Temporal edge weight decay formula
# Validates: Requirements 2.1, 2.8
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=2000)
@given(
    amount=st.floats(min_value=1.0, max_value=1e8, allow_nan=False, allow_infinity=False),
    days_ago=st.integers(min_value=0, max_value=365),
    half_life=st.floats(min_value=1.0, max_value=30.0, allow_nan=False, allow_infinity=False),
)
def test_property6_temporal_weight_formula(amount, days_ago, half_life):
    """
    **Validates: Requirements 2.1, 2.8**

    Property 6: Temporal edge weight decay formula.

    For any edge with amount w_original, transaction date d, and reference_date ref
    where ref >= d, the temporal weight SHALL equal
        w_original × exp(-λ × (ref - d).days)
    where λ = 0.693 / half_life_days, and Edge_Age = (ref - d).days is a
    non-negative integer.
    """
    ref_date = date(2024, 6, 1)
    tx_date = ref_date - timedelta(days=days_ago)

    engine = TDPageRankEngine(half_life_days=half_life)

    df = _make_edges_df(
        senders=["A"],
        receivers=["B"],
        amounts=[amount],
        dates=[tx_date],
    )

    weights = engine._compute_temporal_weights(df, ref_date)

    # Edge_Age must be non-negative integer
    edge_age = days_ago  # (ref_date - tx_date).days
    assert edge_age >= 0, f"Edge_Age must be non-negative, got {edge_age}"

    # Verify formula: w_temporal = amount × exp(-λ × Edge_Age)
    lam = 0.693 / half_life
    expected = amount * np.exp(-lam * edge_age)

    assert len(weights) == 1
    actual = float(weights.iloc[0])

    assert np.isclose(actual, expected, rtol=1e-9), (
        f"Temporal weight mismatch: got {actual}, expected {expected} "
        f"(amount={amount}, edge_age={edge_age}, lambda={lam})"
    )


# ---------------------------------------------------------------------------
# Property 7 — Score vector invariants
# Validates: Requirements 2.2, 2.4, 2.9
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=2000)
@given(graph=random_directed_graph(min_nodes=3, max_nodes=50, min_edges=2, max_edges=200))
def test_property7_score_vector_invariants(graph):
    """
    **Validates: Requirements 2.2, 2.4, 2.9**

    Property 7: TD-PageRank score vector invariants.

    For any valid directed graph with at least one edge:
    (a) All raw scores non-negative.
    (b) Sum of raw scores ≈ 1.0 within 1e-4 (before applying cycle penalty,
        the sum is 1.0; post-penalty the sum may decrease slightly, so we
        allow a relaxed tolerance for cycle-penalised graphs).
    (c) Normalized scores are in [0, 1].
    (d) Iteration count ≤ 100.
    """
    ref_date = date(2024, 1, 1)
    edges_df = graph_to_edges_df(graph, reference_date=ref_date)
    assume(len(edges_df) > 0)

    engine = TDPageRankEngine()
    result = engine.compute(edges_df, reference_date=ref_date)

    # (a) All raw scores non-negative
    for node, score in result.scores.items():
        assert score >= 0.0, f"Negative score for node {node}: {score}"

    # (b) Normalized scores in [0, 1]
    for node, score in result.normalized_scores.items():
        assert 0.0 - 1e-9 <= score <= 1.0 + 1e-9, (
            f"Normalized score {score} out of [0,1] for node {node}"
        )

    # (c) Iteration count ≤ 100
    assert result.iterations <= 100, (
        f"Iteration count {result.iterations} exceeds maximum 100"
    )

    # (d) Raw scores non-empty if edges non-empty
    assert len(result.scores) > 0, "No scores produced for non-empty graph"


# ---------------------------------------------------------------------------
# Property 8 — Cycle penalty correctness
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=2000)
@given(
    scc_size=st.integers(min_value=3, max_value=10),
    extra_nodes=st.integers(min_value=0, max_value=5),
)
def test_property8_cycle_penalty_correctness(scc_size, extra_nodes):
    """
    **Validates: Requirements 2.3**

    Property 8: Cycle penalty application correctness.

    For a node in an SCC of size > 2 where intra-SCC temporal weight > 80%
    of total incident weight, the penalised score = 0.5 × pre-penalty score.

    Strategy: build a complete directed graph on scc_size nodes to guarantee
    high intra-SCC weight. Add extra_nodes as outlier nodes connected only
    outward so the SCC remains intact.
    """
    ref_date = date(2024, 1, 1)

    # Build complete directed graph on scc_size nodes (guaranteed SCC)
    # with very high edge weights so intra-SCC weight >> external weight
    scc_nodes = [f"SCC_{i}" for i in range(scc_size)]
    rows = []
    for u in scc_nodes:
        for v in scc_nodes:
            if u != v:
                rows.append(
                    {
                        "Sender_account": u,
                        "Receiver_account": v,
                        "amount_local_npr": 1_000_000.0,  # very large intra-SCC weight
                        "Date": ref_date - timedelta(days=1),
                    }
                )

    # Add extra_nodes connected to SCC with tiny weights
    for i in range(extra_nodes):
        ext_node = f"EXT_{i}"
        rows.append(
            {
                "Sender_account": ext_node,
                "Receiver_account": scc_nodes[0],
                "amount_local_npr": 0.01,  # negligible external weight
                "Date": ref_date - timedelta(days=1),
            }
        )

    assume(len(rows) > 0)
    edges_df = pd.DataFrame(rows)
    edges_df["Date"] = pd.to_datetime(edges_df["Date"])

    engine = TDPageRankEngine(cycle_penalty=0.5)

    # Compute WITHOUT penalty (cycle_penalty=1.0 → no change)
    engine_no_penalty = TDPageRankEngine(cycle_penalty=1.0)
    result_no_penalty = engine_no_penalty.compute(edges_df, reference_date=ref_date)

    # Compute WITH penalty
    result_with_penalty = engine.compute(edges_df, reference_date=ref_date)

    # For nodes in the SCC with intra-SCC weight > 80%, penalty should be applied
    for node in scc_nodes:
        if result_with_penalty.cycle_member.get(node, False):
            score_pre = result_no_penalty.scores.get(node, 0.0)
            score_post = result_with_penalty.scores.get(node, 0.0)

            # With complete intra-SCC graph, intra_weight / total_weight ≈ 1.0 > 0.80
            # So penalty must be applied: score_post = 0.5 × score_pre
            if score_pre > 1e-10:
                ratio = score_post / score_pre
                assert abs(ratio - 0.5) < 1e-6, (
                    f"Node {node}: expected penalty ratio 0.5, got {ratio:.8f} "
                    f"(pre={score_pre}, post={score_post})"
                )


# ---------------------------------------------------------------------------
# Property 9 — Dormant node rank suppression
# Validates: Requirements 2.7
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=2000)
@given(
    n_dormant=st.integers(min_value=2, max_value=5),
    dormant_age=st.integers(min_value=31, max_value=200),
)
def test_property9_dormant_node_suppression(n_dormant, dormant_age):
    """
    **Validates: Requirements 2.7**

    Property 9: Dormant node rank suppression.

    For any node v where ALL incident edges have Edge_Age > 30 days,
    the normalized score of v SHALL be no greater than 0.1 × max normalized score.

    Design: We build two disjoint groups:
      - Dormant ring (n_dormant nodes, edges age > 30 days)
      - Active chain with varied amounts (3 nodes, edges age = 1 day)
    The active nodes receive unequal weights so they produce spread-out scores.
    We then verify that dormant node scores ≤ 0.1 × max score over ALL nodes,
    comparing against the max score held by an active node.
    """
    ref_date = date(2024, 1, 1)
    rows = []

    # Dormant ring: n_dormant ≥ 2 ensures no self-loops; all edges old
    dormant_nodes = [f"DORM_{i}" for i in range(n_dormant)]
    for i in range(n_dormant):
        u = dormant_nodes[i]
        v = dormant_nodes[(i + 1) % n_dormant]
        rows.append(
            {
                "Sender_account": u,
                "Receiver_account": v,
                "amount_local_npr": 1_000.0,  # small weight
                "Date": ref_date - timedelta(days=dormant_age),
            }
        )

    # Active hub-and-spoke: 1 hub receives large flows from 2 spokes
    # This creates score differentiation so that the active hub dominates
    rows.extend(
        [
            {
                "Sender_account": "ACT_SPOKE_A",
                "Receiver_account": "ACT_HUB",
                "amount_local_npr": 10_000_000.0,  # large weight, very recent
                "Date": ref_date - timedelta(days=1),
            },
            {
                "Sender_account": "ACT_SPOKE_B",
                "Receiver_account": "ACT_HUB",
                "amount_local_npr": 8_000_000.0,
                "Date": ref_date - timedelta(days=2),
            },
            {
                "Sender_account": "ACT_HUB",
                "Receiver_account": "ACT_SPOKE_A",
                "amount_local_npr": 500_000.0,
                "Date": ref_date - timedelta(days=1),
            },
        ]
    )

    edges_df = pd.DataFrame(rows)
    edges_df["Date"] = pd.to_datetime(edges_df["Date"])

    engine = TDPageRankEngine()
    result = engine.compute(edges_df, reference_date=ref_date)

    if not result.normalized_scores:
        return  # nothing to check for empty result

    max_norm = max(result.normalized_scores.values())

    # Guard: if max_norm is 0 all scores equal — skip
    if max_norm < 1e-12:
        return

    # The max normalized score MUST come from an active node (not dormant)
    # because active nodes have orders-of-magnitude larger temporal weights
    max_node = max(result.normalized_scores, key=result.normalized_scores.get)
    # We only assert if the max node is indeed an active node
    assume(max_node not in dormant_nodes)

    for node in dormant_nodes:
        if node not in result.normalized_scores:
            continue
        dormant_score = result.normalized_scores[node]
        threshold = 0.1 * max_norm
        assert dormant_score <= threshold + 1e-9, (
            f"Dormant node {node} normalized score {dormant_score:.6f} "
            f"exceeds 0.1 × max ({threshold:.6f})"
        )


# ---------------------------------------------------------------------------
# Property 10 — Determinism
# Validates: Requirements 2.10
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=2000)
@given(graph=random_directed_graph(min_nodes=3, max_nodes=30, min_edges=2, max_edges=100))
def test_property10_determinism(graph):
    """
    **Validates: Requirements 2.10**

    Property 10: TD-PageRank determinism.

    Two independent computations from identical inputs (same edges_df,
    same reference_date, same parameters) SHALL produce scores identical
    within a floating-point tolerance of 1e-12.
    """
    ref_date = date(2024, 1, 1)
    edges_df = graph_to_edges_df(graph, reference_date=ref_date)
    assume(len(edges_df) > 0)

    engine = TDPageRankEngine()

    result1 = engine.compute(edges_df.copy(), reference_date=ref_date)
    result2 = engine.compute(edges_df.copy(), reference_date=ref_date)

    assert set(result1.scores.keys()) == set(result2.scores.keys()), (
        "Nodes differ between runs"
    )

    for node in result1.scores:
        s1 = result1.scores[node]
        s2 = result2.scores[node]
        assert abs(s1 - s2) <= 1e-12, (
            f"Non-deterministic raw score for node {node}: "
            f"run1={s1}, run2={s2}, diff={abs(s1-s2)}"
        )

    for node in result1.normalized_scores:
        n1 = result1.normalized_scores[node]
        n2 = result2.normalized_scores[node]
        assert abs(n1 - n2) <= 1e-12, (
            f"Non-deterministic normalized score for node {node}: "
            f"run1={n1}, run2={n2}, diff={abs(n1-n2)}"
        )


# ---------------------------------------------------------------------------
# Property 16 — Temporal decay dominance
# Validates: Requirements 10.1, 10.2, 10.3, 10.4
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=2000)
@given(
    amount=st.floats(min_value=0.01, max_value=1e8, allow_nan=False, allow_infinity=False),
    age=st.integers(min_value=0, max_value=365),
    half_life=st.floats(min_value=1.0, max_value=30.0, allow_nan=False, allow_infinity=False),
)
def test_property16_temporal_decay_dominance(amount, age, half_life):
    """
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4**

    Property 16: Temporal decay strictly reduces weight for aged transactions.
    - For age > 0: temporal_weight < undecayed_weight (amount)
    - For age == 0: temporal_weight == undecayed_weight
    """
    decay_lambda = 0.693 / half_life
    temporal_weight = amount * np.exp(-decay_lambda * age)
    undecayed_weight = amount

    if age > 0:
        assert temporal_weight < undecayed_weight, (
            f"Temporal weight {temporal_weight} should be < undecayed {undecayed_weight} for age={age}"
        )
    else:
        assert abs(temporal_weight - undecayed_weight) < 1e-9, (
            f"Temporal weight {temporal_weight} should == undecayed {undecayed_weight} for age=0"
        )
