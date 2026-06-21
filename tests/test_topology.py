"""
Property-based tests for EgoNetworkExtractor / TopologyVector.

Properties tested:
    Property 1  (Task 4.2) – TopologyVector completeness and correctness

Uses Hypothesis and the shared random_directed_graph strategy from conftest.
"""

import sys
import os

import networkx as nx
import pytest
from hypothesis import given, settings, assume, strategies as st

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.adaptive_fusion import EgoNetworkExtractor
from tests.conftest import random_directed_graph


# ---------------------------------------------------------------------------
# Property 1 — TopologyVector completeness and correctness
# Validates: Requirements 1.1
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=2000)
@given(
    graph=random_directed_graph(min_nodes=5, max_nodes=100, min_edges=1, max_edges=300)
)
def test_property1_topology_vector_completeness(graph):
    """
    **Validates: Requirements 1.1**

    Property 1: Topology Vector completeness and correctness.

    For any directed graph G with at least 1 edge and any node v in G,
    the TopologyVector computed from v's 2-hop ego-network SHALL:
      - contain exactly 5 metrics (edge_density, diameter, avg_clustering,
        degree_asymmetry, component_ratio)
      - edge_density ∈ [0, 1]
      - diameter ≥ 0
      - avg_clustering ∈ [0, 1]
      - degree_asymmetry ≥ 0
      - component_ratio ∈ (0, 1]
    """
    assume(graph.number_of_edges() >= 1)

    nodes = list(graph.nodes())
    assume(len(nodes) >= 1)

    # Pick the first node as target (deterministic, no draw needed)
    account_id = nodes[0]

    extractor = EgoNetworkExtractor()
    tv = extractor.compute_topology_vector(graph, account_id, max_hops=2, timeout_ms=500.0)

    # Result must not be None for a graph that contains account_id
    assert tv is not None, (
        f"TopologyVector returned None for account '{account_id}' "
        f"which is in the graph (nodes={len(nodes)}, edges={graph.number_of_edges()})"
    )

    # Check 5 fields exist (dataclass guarantees this, but verify via attribute access)
    fields = ["edge_density", "diameter", "avg_clustering", "degree_asymmetry", "component_ratio"]
    for field_name in fields:
        assert hasattr(tv, field_name), f"TopologyVector missing field: {field_name}"

    # edge_density ∈ [0, 1]
    assert 0.0 - 1e-9 <= tv.edge_density <= 1.0 + 1e-9, (
        f"edge_density {tv.edge_density} not in [0, 1]"
    )

    # diameter ≥ 0
    assert tv.diameter >= 0, f"diameter {tv.diameter} is negative"

    # avg_clustering ∈ [0, 1]
    assert 0.0 - 1e-9 <= tv.avg_clustering <= 1.0 + 1e-9, (
        f"avg_clustering {tv.avg_clustering} not in [0, 1]"
    )

    # degree_asymmetry ≥ 0
    assert tv.degree_asymmetry >= 0.0 - 1e-9, (
        f"degree_asymmetry {tv.degree_asymmetry} is negative"
    )

    # component_ratio ∈ (0, 1]
    assert tv.component_ratio > 0.0 - 1e-9, (
        f"component_ratio {tv.component_ratio} must be > 0"
    )
    assert tv.component_ratio <= 1.0 + 1e-9, (
        f"component_ratio {tv.component_ratio} must be ≤ 1"
    )


@settings(max_examples=20, deadline=2000)
@given(
    graph=random_directed_graph(min_nodes=5, max_nodes=50, min_edges=1, max_edges=100),
    node_idx=st.integers(min_value=0, max_value=49),
)
def test_property1_topology_vector_completeness_random_node(graph, node_idx):
    """
    **Validates: Requirements 1.1**

    Supplementary variant: test against a random target node (not just the first).
    """
    assume(graph.number_of_edges() >= 1)

    nodes = list(graph.nodes())
    assume(len(nodes) >= 1)

    # Pick any valid node by index
    account_id = nodes[node_idx % len(nodes)]

    extractor = EgoNetworkExtractor()
    tv = extractor.compute_topology_vector(graph, account_id, max_hops=2, timeout_ms=500.0)

    # account_id is guaranteed to be in the graph
    assert tv is not None

    assert 0.0 - 1e-9 <= tv.edge_density <= 1.0 + 1e-9
    assert tv.diameter >= 0
    assert 0.0 - 1e-9 <= tv.avg_clustering <= 1.0 + 1e-9
    assert tv.degree_asymmetry >= 0.0 - 1e-9
    assert 0.0 < tv.component_ratio <= 1.0 + 1e-9


def test_topology_vector_account_not_in_graph():
    """Unit test: account_id not in graph → returns None."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    extractor = EgoNetworkExtractor()
    tv = extractor.compute_topology_vector(G, "UNKNOWN", max_hops=2)
    assert tv is None


def test_topology_vector_isolated_node():
    """Unit test: isolated node (< 3 nodes in ego-network) → sparse but valid vector."""
    G = nx.DiGraph()
    G.add_node("SOLO")
    G.add_edge("A", "B")  # unrelated edge so graph is non-trivial

    extractor = EgoNetworkExtractor()
    tv = extractor.compute_topology_vector(G, "SOLO", max_hops=2)

    assert tv is not None
    assert tv.edge_density == 0.0
    assert tv.component_ratio == 1.0


def test_topology_vector_to_tensor():
    """Unit test: to_tensor() returns a 1-D float tensor of length 5."""
    import torch
    G = nx.DiGraph()
    for i in range(5):
        G.add_edge(f"N{i}", f"N{(i+1) % 5}", amount_local_npr=1000.0, days_ago=5)

    extractor = EgoNetworkExtractor()
    tv = extractor.compute_topology_vector(G, "N0", max_hops=2, timeout_ms=500.0)

    assert tv is not None
    tensor = tv.to_tensor()
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (5,)
    assert tensor.dtype == torch.float32
