"""
Shared fixtures and Hypothesis custom strategies for AML innovation tests.

Provides reusable strategies for:
- Random directed graphs (for TD-PageRank and topology tests)
- Topology vectors (5 metrics)
- Health vector sequences (for degradation controller tests)
- Temporal edge DataFrames with columns [Sender_account, Receiver_account, amount_local_npr, Date]
- SCCFlowFeatures (for learnable SCC penalty property tests)
- Ego-networks as PyG Data objects (for topology embedding property tests)
- Precision estimate sequences (for online precision monitor property tests)
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import networkx as nx
import torch
import pytest
from hypothesis import strategies as st, settings

# Add project root to sys.path for cross-module imports
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.hardening_data_models import SCCFlowFeatures


# ---------------------------------------------------------------------------
# Hypothesis Global Settings
# ---------------------------------------------------------------------------
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=1000,
    derandomize=True,
)
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=2000,
    derandomize=False,
)
settings.load_profile("ci")


# ---------------------------------------------------------------------------
# Hypothesis Custom Strategies
# ---------------------------------------------------------------------------


@st.composite
def random_directed_graph(draw, min_nodes=3, max_nodes=50, min_edges=2, max_edges=200):
    """
    Generate random directed graphs for TD-PageRank and topology tests.

    Returns a NetworkX DiGraph with edge attributes:
    - 'amount_local_npr': float transaction amount
    - 'days_ago': int days before reference date

    Nodes are labeled as strings (e.g., "ACC_0", "ACC_1") to match
    the account ID format used in the real system.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    max_possible_edges = n * (n - 1)
    m = draw(st.integers(min_value=min_edges, max_value=min(max_edges, max_possible_edges)))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))

    G = nx.gnm_random_graph(n, m, directed=True, seed=seed)

    # Relabel nodes to account-style string IDs
    mapping = {i: f"ACC_{i}" for i in range(n)}
    G = nx.relabel_nodes(G, mapping)

    # Add edge attributes
    for u, v in G.edges():
        G[u][v]["amount_local_npr"] = draw(
            st.floats(min_value=100.0, max_value=1e7, allow_nan=False, allow_infinity=False)
        )
        G[u][v]["days_ago"] = draw(st.integers(min_value=0, max_value=365))

    return G


@st.composite
def topology_vector(draw):
    """
    Generate random TopologyVector-compatible tuples (5 metrics).

    Returns a dict with keys matching TopologyVector fields:
    - edge_density: float in [0, 1]
    - diameter: non-negative int
    - avg_clustering: float in [0, 1]
    - degree_asymmetry: non-negative float
    - component_ratio: float in (0, 1]
    """
    return {
        "edge_density": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        "diameter": draw(st.integers(min_value=0, max_value=10)),
        "avg_clustering": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        "degree_asymmetry": draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        "component_ratio": draw(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
    }


@st.composite
def health_vector_sequence(draw, min_length=2, max_length=20):
    """
    Generate sequences of HealthVector observations for degradation controller tests.

    Each entry is a dict with:
    - heartbeat_latency_ms: float in [0, 10000]
    - kl_divergence: float in [0, 2.0]
    - throughput_ratio: float in [0, 2.0]
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    return [
        {
            "heartbeat_latency_ms": draw(
                st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
            ),
            "kl_divergence": draw(
                st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
            ),
            "throughput_ratio": draw(
                st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
            ),
        }
        for _ in range(length)
    ]


# ---------------------------------------------------------------------------
# Architecture Hardening: Custom Hypothesis Strategies
# ---------------------------------------------------------------------------


# Strategy for generating random SCCFlowFeatures
scc_flow_features = st.builds(
    SCCFlowFeatures,
    intra_inflow_weight=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    intra_outflow_weight=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    weight_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    scc_size=st.integers(min_value=3, max_value=50),
    node_degree_in_scc=st.integers(min_value=1, max_value=49),
)


def generate_random_ego_network(num_nodes: int, edge_density: float):
    """
    Generate a random ego-network as a PyTorch Geometric Data object.

    Args:
        num_nodes: Number of nodes in the ego-network (1 to 100).
        edge_density: Proportion of possible edges present (0.0 to 1.0).

    Returns:
        torch_geometric.data.Data with:
        - x: Node feature matrix [num_nodes, 4] with [in_degree, out_degree, amount_sum, edge_count]
        - edge_index: COO format edge tensor [2, num_edges]
    """
    try:
        import torch_geometric
        from torch_geometric.data import Data
    except ImportError:
        # Fallback: return a minimal Data-like object if PyG not available
        raise ImportError("torch_geometric is required for ego_networks strategy")

    if num_nodes == 0:
        # Empty ego-network
        return Data(
            x=torch.zeros((0, 4), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )

    # Generate directed graph
    max_edges = num_nodes * (num_nodes - 1)
    num_edges = int(edge_density * max_edges)

    if num_edges > 0 and num_nodes > 1:
        # Generate random edge indices (no self-loops)
        all_possible = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    all_possible.append((i, j))

        # Sample edges
        rng = np.random.default_rng(seed=num_nodes * 1000 + int(edge_density * 1000))
        num_edges = min(num_edges, len(all_possible))
        if num_edges > 0:
            indices = rng.choice(len(all_possible), size=num_edges, replace=False)
            edges = [all_possible[idx] for idx in indices]
            src = [e[0] for e in edges]
            dst = [e[1] for e in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Compute node features: [in_degree, out_degree, amount_sum, edge_count]
    in_degrees = torch.zeros(num_nodes, dtype=torch.float32)
    out_degrees = torch.zeros(num_nodes, dtype=torch.float32)

    if edge_index.shape[1] > 0:
        for i in range(edge_index.shape[1]):
            out_degrees[edge_index[0, i]] += 1
            in_degrees[edge_index[1, i]] += 1

    # amount_sum and edge_count are synthetic features
    amount_sum = (in_degrees + out_degrees) * 1000.0  # Proxy for total amount
    edge_count = in_degrees + out_degrees

    x = torch.stack([in_degrees, out_degrees, amount_sum, edge_count], dim=1)

    return Data(x=x, edge_index=edge_index)


# Strategy for generating random ego-networks as PyG Data objects
ego_networks = st.builds(
    generate_random_ego_network,
    num_nodes=st.integers(min_value=1, max_value=100),
    edge_density=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


# Strategy for generating precision estimate sequences
precision_sequences = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=50,
)


# ---------------------------------------------------------------------------
# Helper: Convert directed graph to temporal edge DataFrame
# ---------------------------------------------------------------------------


def graph_to_edges_df(G: nx.DiGraph, reference_date: date = None) -> pd.DataFrame:
    """
    Convert a NetworkX DiGraph (with 'amount_local_npr' and 'days_ago' attributes)
    to a temporal edge DataFrame with columns:
    [Sender_account, Receiver_account, amount_local_npr, Date]

    Args:
        G: Directed graph with edge attributes from random_directed_graph strategy
        reference_date: The reference date from which 'days_ago' is subtracted.
                       Defaults to 2024-01-01 if not provided.

    Returns:
        DataFrame with the required schema for TD-PageRank engine.
    """
    if reference_date is None:
        reference_date = date(2024, 1, 1)

    rows = []
    for u, v, data in G.edges(data=True):
        days_ago = data.get("days_ago", 0)
        tx_date = reference_date - timedelta(days=days_ago)
        rows.append(
            {
                "Sender_account": u,
                "Receiver_account": v,
                "amount_local_npr": data.get("amount_local_npr", 1000.0),
                "Date": tx_date,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["Sender_account", "Receiver_account", "amount_local_npr", "Date"]
        )

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------------------------------------------------------------------------
# Shared Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_edges_df():
    """
    A small, deterministic temporal edge DataFrame for unit tests.

    Columns: [Sender_account, Receiver_account, amount_local_npr, Date]
    Contains 6 transactions forming a small directed graph with
    a mix of recent and old edges.
    """
    reference_date = date(2024, 1, 1)
    data = [
        {"Sender_account": "ACC_0", "Receiver_account": "ACC_1", "amount_local_npr": 50000.0, "Date": reference_date - timedelta(days=2)},
        {"Sender_account": "ACC_1", "Receiver_account": "ACC_2", "amount_local_npr": 30000.0, "Date": reference_date - timedelta(days=5)},
        {"Sender_account": "ACC_2", "Receiver_account": "ACC_0", "amount_local_npr": 25000.0, "Date": reference_date - timedelta(days=3)},
        {"Sender_account": "ACC_0", "Receiver_account": "ACC_3", "amount_local_npr": 100000.0, "Date": reference_date - timedelta(days=1)},
        {"Sender_account": "ACC_3", "Receiver_account": "ACC_4", "amount_local_npr": 75000.0, "Date": reference_date - timedelta(days=10)},
        {"Sender_account": "ACC_4", "Receiver_account": "ACC_1", "amount_local_npr": 20000.0, "Date": reference_date - timedelta(days=45)},
    ]
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture
def sample_directed_graph():
    """
    A small, deterministic directed graph for unit tests.

    5 nodes (ACC_0 through ACC_4) with 6 edges.
    Includes a 3-node cycle (ACC_0 → ACC_1 → ACC_2 → ACC_0)
    and a tail (ACC_0 → ACC_3 → ACC_4 → ACC_1).
    """
    G = nx.DiGraph()
    G.add_nodes_from(["ACC_0", "ACC_1", "ACC_2", "ACC_3", "ACC_4"])
    edges = [
        ("ACC_0", "ACC_1", {"amount_local_npr": 50000.0, "days_ago": 2}),
        ("ACC_1", "ACC_2", {"amount_local_npr": 30000.0, "days_ago": 5}),
        ("ACC_2", "ACC_0", {"amount_local_npr": 25000.0, "days_ago": 3}),
        ("ACC_0", "ACC_3", {"amount_local_npr": 100000.0, "days_ago": 1}),
        ("ACC_3", "ACC_4", {"amount_local_npr": 75000.0, "days_ago": 10}),
        ("ACC_4", "ACC_1", {"amount_local_npr": 20000.0, "days_ago": 45}),
    ]
    G.add_edges_from(edges)
    return G


@pytest.fixture
def reference_date():
    """Default reference date for tests (2024-01-01)."""
    return date(2024, 1, 1)


@pytest.fixture
def dormant_edges_df():
    """
    Edge DataFrame where ALL edges are older than 30 days.
    Used for testing dormant node rank suppression (Property 9).
    """
    reference_date = date(2024, 1, 1)
    data = [
        {"Sender_account": "ACC_0", "Receiver_account": "ACC_1", "amount_local_npr": 50000.0, "Date": reference_date - timedelta(days=60)},
        {"Sender_account": "ACC_1", "Receiver_account": "ACC_2", "amount_local_npr": 30000.0, "Date": reference_date - timedelta(days=90)},
        {"Sender_account": "ACC_2", "Receiver_account": "ACC_0", "amount_local_npr": 25000.0, "Date": reference_date - timedelta(days=45)},
        # Add a recent node (ACC_3) to contrast with dormant nodes
        {"Sender_account": "ACC_3", "Receiver_account": "ACC_4", "amount_local_npr": 100000.0, "Date": reference_date - timedelta(days=1)},
        {"Sender_account": "ACC_4", "Receiver_account": "ACC_3", "amount_local_npr": 80000.0, "Date": reference_date - timedelta(days=2)},
    ]
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df
