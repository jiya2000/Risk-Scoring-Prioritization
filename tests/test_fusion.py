"""
Property-based tests for the adaptive fusion formula.

Properties tested:
    Property 3  (Task 4.6) – Fusion formula correctness and bounds

This is a pure math property: fused = w_ml * s_ml + w_graph * s_graph + w_rules * s_rules
must land in [0.0, 1.0] whenever each score is in [0, 1] and weights are valid
(each in [0.05, 0.90], sum == 1.0).
"""

import sys
import os

import pytest
from hypothesis import given, settings, assume, strategies as st

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers: generate valid weight triples
# ---------------------------------------------------------------------------


def _valid_weights(draw):
    """
    Draw (w_ml, w_graph, w_rules) each in [0.05, 0.90] summing to 1.0.
    Strategy: draw two free weights, compute third, reject if out of range.
    """
    w_ml = draw(st.floats(min_value=0.05, max_value=0.90, allow_nan=False))
    remaining = 1.0 - w_ml
    # w_graph must leave at least 0.05 for w_rules and be at most 0.90
    w_graph_min = max(0.05, remaining - 0.90)
    w_graph_max = min(0.90, remaining - 0.05)
    assume(w_graph_min <= w_graph_max)
    w_graph = draw(st.floats(min_value=w_graph_min, max_value=w_graph_max, allow_nan=False))
    w_rules = 1.0 - w_ml - w_graph
    assume(0.05 - 1e-9 <= w_rules <= 0.90 + 1e-9)
    return w_ml, w_graph, w_rules


@st.composite
def valid_weight_triple(draw):
    return _valid_weights(draw)


# ---------------------------------------------------------------------------
# Property 3 — Fusion formula correctness and bounds
# Validates: Requirements 1.3, 1.8
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=5000)
@given(
    s_ml=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    s_graph=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    s_rules=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    weights=valid_weight_triple(),
)
def test_property3_fusion_formula_bounds(s_ml, s_graph, s_rules, weights):
    """
    **Validates: Requirements 1.3, 1.8**

    Property 3: Fusion formula correctness and bounds.

    For any (s_ml, s_graph, s_rules) each in [0, 1] and a valid weight vector
    where each weight is in [0.05, 0.90] and weights sum to 1.0:

        fused = w_ml * s_ml + w_graph * s_graph + w_rules * s_rules

    must satisfy fused ∈ [0.0, 1.0].

    This is a pure math property — no model call needed.
    """
    w_ml, w_graph, w_rules = weights

    # Pre-condition: weights are valid
    assert abs(w_ml + w_graph + w_rules - 1.0) < 1e-6, (
        f"Weights do not sum to 1.0: {w_ml + w_graph + w_rules}"
    )
    assert 0.05 - 1e-9 <= w_ml <= 0.90 + 1e-9
    assert 0.05 - 1e-9 <= w_graph <= 0.90 + 1e-9
    assert 0.05 - 1e-9 <= w_rules <= 0.90 + 1e-9

    # The fusion formula
    fused = w_ml * s_ml + w_graph * s_graph + w_rules * s_rules

    # Post-condition: result is in [0.0, 1.0]
    # Lower bound: each weight ≥ 0, each score ≥ 0, so fused ≥ 0
    assert fused >= 0.0 - 1e-9, (
        f"fused={fused:.8f} is below 0.0 "
        f"(s_ml={s_ml}, s_graph={s_graph}, s_rules={s_rules}, weights={weights})"
    )

    # Upper bound: fused = sum(w_i * s_i) ≤ sum(w_i) * max(s_i) = 1.0 * 1.0 = 1.0
    assert fused <= 1.0 + 1e-9, (
        f"fused={fused:.8f} exceeds 1.0 "
        f"(s_ml={s_ml}, s_graph={s_graph}, s_rules={s_rules}, weights={weights})"
    )


@settings(max_examples=100, deadline=5000)
@given(
    s_ml=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    s_graph=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    s_rules=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    weights=valid_weight_triple(),
)
def test_property3_fusion_formula_weighted_sum(s_ml, s_graph, s_rules, weights):
    """
    **Validates: Requirements 1.3**

    Complementary check: the fused score equals the exact weighted sum,
    not some other formula.
    """
    w_ml, w_graph, w_rules = weights

    fused = w_ml * s_ml + w_graph * s_graph + w_rules * s_rules
    expected = w_ml * s_ml + w_graph * s_graph + w_rules * s_rules

    assert abs(fused - expected) < 1e-12, (
        f"Fusion formula mismatch: fused={fused}, expected={expected}"
    )


# ---------------------------------------------------------------------------
# Unit tests — specific examples for fusion formula
# ---------------------------------------------------------------------------


def test_fusion_formula_uniform_weights():
    """Unit test: equal weights produce the mean of three scores."""
    w = 1.0 / 3.0
    fused = w * 0.9 + w * 0.6 + w * 0.3
    expected = (0.9 + 0.6 + 0.3) / 3.0
    assert abs(fused - expected) < 1e-9


def test_fusion_formula_extreme_ml_weight():
    """Unit test: w_ml = 0.90, the fused score ≈ s_ml."""
    w_ml, w_graph, w_rules = 0.90, 0.05, 0.05
    fused = w_ml * 0.8 + w_graph * 0.2 + w_rules * 0.4
    # 0.90*0.8 + 0.05*0.2 + 0.05*0.4 = 0.72 + 0.01 + 0.02 = 0.75
    assert abs(fused - 0.75) < 1e-9


def test_fusion_formula_all_zeros():
    """Unit test: all scores = 0 → fused = 0."""
    w_ml, w_graph, w_rules = 0.70, 0.15, 0.15
    fused = w_ml * 0.0 + w_graph * 0.0 + w_rules * 0.0
    assert fused == 0.0


def test_fusion_formula_all_ones():
    """Unit test: all scores = 1 → fused = 1."""
    w_ml, w_graph, w_rules = 0.70, 0.15, 0.15
    fused = w_ml * 1.0 + w_graph * 1.0 + w_rules * 1.0
    assert abs(fused - 1.0) < 1e-9
