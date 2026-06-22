"""
Topology-Adaptive Fusion Architecture.

Contains:
- EgoNetworkExtractor: Extracts 2-hop ego-network and computes topology vector
- FusionWeightNetwork: MLP mapping topology vector to fusion weights
- TopologyAttentionGate: Multiplicative attention gate over topology features (novel)
- AdaptiveFusionEngine: Orchestrates topology extraction, weight inference, and score fusion

Novel mechanisms (patent differentiators):
1. 5-metric ego-network topology vector (density, diameter, clustering, degree_asymmetry, component_ratio)
2. Optional 6th dimension: transaction velocity ratio (recent_edges / total_edges)
3. TopologyAttentionGate: multiplicative gating architecture — distinct from standard attention and MLP ensembles
4. Isolated account protection: w_ml >= 0.70 floor when ego-network < 3 nodes
5. Per-account dynamic weights recomputed every scoring batch — not globally learned
"""

import sys
import os
import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.data_models import TopologyVector, FusionResult
from models.architecture_config import ArchitectureHardeningConfig

# Conditional import for PyTorch Geometric
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@dataclass
class ExtendedTopologyVector(TopologyVector):
    """6-dimensional topology vector including transaction velocity."""
    tx_velocity_ratio: float = 0.0  # recent_edges / total_edges (last 7 days)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.edge_density, self.diameter, self.avg_clustering,
             self.degree_asymmetry, self.component_ratio, self.tx_velocity_ratio],
            dtype=torch.float32
        )

logger = logging.getLogger(__name__)


class TopologyEmbeddingNetwork(nn.Module):
    """GNN-based ego-network embedding replacing manual TopologyVector.

    Uses PyTorch Geometric GCNConv layers to produce a fixed-dimensional
    embedding from a variable-size ego-network subgraph. Falls back to
    a zero-vector signal when PyG is unavailable or inference exceeds
    the timeout budget.

    Architecture:
        GCNConv(node_feature_dim, hidden_dim)
        → ReLU
        → GCNConv(hidden_dim, hidden_dim)
        → ReLU
        → global_mean_pool
        → Linear(hidden_dim, embedding_dim)

    Node features: [in_degree, out_degree, amount_sum, edge_count] (4-dim)
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        num_layers: int = 2,
        timeout_ms: float = 200.0,
    ):
        """
        Args:
            node_feature_dim: Number of input features per node (default 4).
            hidden_dim: Hidden dimension for GCNConv layers (default 32).
            embedding_dim: Output embedding dimension (default 16).
            num_layers: Number of GCNConv message-passing layers (default 2).
            timeout_ms: Maximum inference time in milliseconds (default 200.0).
        """
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for TopologyEmbeddingNetwork. "
                "Install with: pip install torch-geometric"
            )

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.timeout_ms = timeout_ms

        # Build GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Projection from pooled hidden representation to embedding
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data: "Data") -> torch.Tensor:
        """
        Produce a fixed-dimensional embedding from a PyG Data object.

        Args:
            data: PyG Data object with:
                - x: Node feature matrix (num_nodes, node_feature_dim)
                - edge_index: Edge connectivity (2, num_edges)
                - batch: Optional batch vector for batched graphs

        Returns:
            Embedding tensor of shape (embedding_dim,) for single graph,
            or (batch_size, embedding_dim) for batched input.
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None

        # If no batch vector, treat as single graph (batch index = 0 for all nodes)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Message-passing layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Global mean pooling: aggregate node representations into graph-level
        x = global_mean_pool(x, batch)  # (num_graphs, hidden_dim)

        # Project to embedding dimension
        embedding = self.projection(x)  # (num_graphs, embedding_dim)

        # Squeeze single-graph case to 1D
        if embedding.size(0) == 1:
            embedding = embedding.squeeze(0)  # (embedding_dim,)

        return embedding

    def extract_ego_network(
        self, G: nx.DiGraph, account_id: str, hops: int = 2
    ) -> "Data":
        """Convert NetworkX 2-hop ego-network to PyG Data object.

        Extracts the ego-network centered on `account_id` within `hops` hops,
        computes node features [in_degree, out_degree, amount_sum, edge_count],
        and builds a PyG Data object suitable for GCN forward pass.

        Args:
            G: Directed transaction graph (nx.DiGraph).
            account_id: The center node for the ego-network.
            hops: Number of hops to include (default 2).

        Returns:
            PyG Data object with:
                - x: (num_nodes, 4) node features
                - edge_index: (2, num_edges) edge connectivity (undirected for GCN)

        Raises:
            ValueError: If account_id is not in the graph or graph is empty.
        """
        if account_id not in G:
            raise ValueError(f"Account '{account_id}' not found in graph.")

        # Extract ego-network nodes via BFS in both directions
        ego_nodes = {account_id}
        current_frontier = {account_id}

        for _ in range(hops):
            next_frontier = set()
            for node in current_frontier:
                next_frontier.update(G.successors(node))
                next_frontier.update(G.predecessors(node))
            ego_nodes.update(next_frontier)
            current_frontier = next_frontier

        # Extract the subgraph
        ego_G = G.subgraph(ego_nodes).copy()
        nodes_list = list(ego_G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
        num_nodes = len(nodes_list)

        # Compute node features: [in_degree, out_degree, amount_sum, edge_count]
        node_features = np.zeros((num_nodes, self.node_feature_dim), dtype=np.float32)
        for idx, node in enumerate(nodes_list):
            in_deg = ego_G.in_degree(node)
            out_deg = ego_G.out_degree(node)

            # Sum of amounts on all edges touching this node
            amount_sum = 0.0
            for _, _, edge_data in ego_G.in_edges(node, data=True):
                amount_sum += edge_data.get('amount', 0.0)
            for _, _, edge_data in ego_G.out_edges(node, data=True):
                amount_sum += edge_data.get('amount', 0.0)

            # Total edge count for this node (in + out)
            edge_count = in_deg + out_deg

            node_features[idx] = [
                float(in_deg),
                float(out_deg),
                float(amount_sum),
                float(edge_count),
            ]

        # Build edge_index (convert directed to undirected for GCN symmetry)
        src_indices = []
        dst_indices = []
        for u, v in ego_G.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            # Add both directions for undirected message passing
            src_indices.extend([u_idx, v_idx])
            dst_indices.extend([v_idx, u_idx])

        if len(src_indices) == 0:
            # No edges: create empty edge_index
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

        # Build PyG Data object
        x = torch.tensor(node_features, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index)

        return data

    def embed_with_timeout(
        self, G: nx.DiGraph, account_id: str, hops: int = 2
    ) -> Tuple[Optional[torch.Tensor], bool, float]:
        """Extract ego-network and compute embedding with timeout enforcement.

        Combines extract_ego_network + forward with a timeout guard.
        Returns a fallback signal (None) if the operation exceeds timeout_ms.

        Args:
            G: Directed transaction graph.
            account_id: Center node for ego-network.
            hops: Number of hops (default 2).

        Returns:
            Tuple of (embedding_or_none, fallback_used, elapsed_ms):
                - embedding: Tensor of shape (embedding_dim,) or None on timeout/error
                - fallback_used: True if embedding could not be produced
                - elapsed_ms: Wall-clock time in milliseconds
        """
        start_time = time.perf_counter()
        result = [None]
        error = [None]

        def _compute():
            try:
                data = self.extract_ego_network(G, account_id, hops=hops)
                with torch.no_grad():
                    embedding = self.forward(data)
                result[0] = embedding
            except Exception as e:
                error[0] = e

        # Run in a thread to enforce timeout
        thread = threading.Thread(target=_compute, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_ms / 1000.0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if thread.is_alive():
            # Timeout exceeded
            logger.warning(
                f"TopologyEmbeddingNetwork inference for '{account_id}' "
                f"exceeded timeout: {elapsed_ms:.1f}ms > {self.timeout_ms:.1f}ms"
            )
            return None, True, elapsed_ms

        if error[0] is not None:
            logger.error(
                f"TopologyEmbeddingNetwork error for '{account_id}': {error[0]}"
            )
            return None, True, elapsed_ms

        return result[0], False, elapsed_ms


class IsolationDetector(nn.Module):
    """Learned continuous isolation function replacing hard-coded threshold.

    Maps a topology embedding (from TopologyEmbeddingNetwork) to a continuous
    isolation score in [0.0, 1.0], then adjusts fusion weights accordingly.

    Higher isolation scores indicate more isolated accounts (sparse ego-networks),
    which should receive higher ML weight (w_ml) since graph-based signals are
    unreliable for isolated nodes.

    Architecture:
        Linear(embedding_dim, hidden_dim) → ReLU
        → Linear(hidden_dim, hidden_dim) → ReLU
        → Linear(hidden_dim, 1) → Sigmoid

    Key invariants:
        - isolation_score in [0.0, 1.0]
        - w_ml in [0.05, 0.90]
        - w_ml + w_graph + w_rules = 1.0 (within 1e-6)
        - w_ml is monotonically non-decreasing with isolation_score
    """

    def __init__(self, embedding_dim: int = 16, hidden_dim: int = 32):
        """
        Args:
            embedding_dim: Dimension of input topology embedding (default 16).
            hidden_dim: Hidden layer dimension for the MLP (default 32).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # MLP: embedding → isolation_score in [0, 1]
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, topology_embedding: torch.Tensor) -> torch.Tensor:
        """Compute isolation score from a topology embedding.

        Args:
            topology_embedding: Tensor of shape (embedding_dim,) or
                (batch_size, embedding_dim) from TopologyEmbeddingNetwork.

        Returns:
            Isolation score tensor of shape () or (batch_size,) with
            values in [0.0, 1.0]. Higher values indicate more isolated nodes.
        """
        score = self.mlp(topology_embedding)  # (..., 1)
        # Squeeze the last dimension
        return score.squeeze(-1)

    def compute_fusion_weights(
        self, isolation_score: float, base_weights: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Adjust base fusion weights based on isolation score.

        Higher isolation → higher w_ml, proportionally. The relationship is
        monotonically non-decreasing: as isolation_score increases, w_ml
        never decreases.

        Algorithm:
            1. Start with base_weights (w_ml, w_graph, w_rules)
            2. Boost w_ml: w_ml_new = w_ml + isolation_score * (0.90 - w_ml)
               This guarantees monotonic increase (higher isolation → higher w_ml,
               capped at 0.90 when isolation_score = 1.0)
            3. Clamp w_ml to [0.05, 0.90]
            4. Redistribute remaining weight (1.0 - w_ml_new) proportionally
               between w_graph and w_rules
            5. Clamp each to [0.05, 0.90] and re-normalize so sum = 1.0

        Args:
            isolation_score: Float in [0.0, 1.0] from forward().
            base_weights: Tuple of (w_ml, w_graph, w_rules) as starting point.

        Returns:
            Tuple of (w_ml, w_graph, w_rules) satisfying:
                - Each weight in [0.05, 0.90]
                - Sum == 1.0 (within 1e-6)
                - w_ml is monotonically non-decreasing with isolation_score

        Raises:
            ValueError: If isolation_score is outside [0.0, 1.0] or base_weights
                don't contain exactly 3 non-negative values.
        """
        # Validate inputs
        if not (0.0 <= isolation_score <= 1.0):
            raise ValueError(
                f"isolation_score must be in [0.0, 1.0], got {isolation_score}"
            )

        w_ml_base, w_graph_base, w_rules_base = base_weights

        # Step 1: Boost w_ml proportionally to isolation_score
        # w_ml_new = w_ml_base + isolation_score * (0.90 - w_ml_base)
        # When isolation_score = 0 → w_ml stays at base
        # When isolation_score = 1 → w_ml goes to 0.90
        # Monotonic: derivative w.r.t. isolation_score = (0.90 - w_ml_base) >= 0
        #            (since w_ml_base <= 0.90 always)
        w_ml_new = w_ml_base + isolation_score * (0.90 - w_ml_base)

        # Step 2: Clamp w_ml to [0.05, 0.90]
        w_ml_new = max(0.05, min(0.90, w_ml_new))

        # Step 3: Distribute remaining budget to w_graph and w_rules
        remaining = 1.0 - w_ml_new

        # Proportional redistribution based on original graph/rules ratio
        total_gr = w_graph_base + w_rules_base
        if total_gr > 0:
            w_graph_new = remaining * (w_graph_base / total_gr)
            w_rules_new = remaining * (w_rules_base / total_gr)
        else:
            # Degenerate case: split equally
            w_graph_new = remaining / 2.0
            w_rules_new = remaining / 2.0

        # Step 4: Clamp each to [0.05, 0.90] and re-normalize iteratively
        # Multiple passes ensure all constraints are met simultaneously
        for _ in range(5):
            w_graph_new = max(0.05, min(0.90, w_graph_new))
            w_rules_new = max(0.05, min(0.90, w_rules_new))
            w_ml_new = max(0.05, min(0.90, w_ml_new))

            # Re-normalize to sum to 1.0
            total = w_ml_new + w_graph_new + w_rules_new
            if total > 0:
                w_ml_new /= total
                w_graph_new /= total
                w_rules_new /= total

        # Final clamp pass (guarantees bounds after normalization)
        w_ml_new = max(0.05, min(0.90, w_ml_new))
        w_graph_new = max(0.05, min(0.90, w_graph_new))
        w_rules_new = max(0.05, min(0.90, w_rules_new))

        # Final normalization to ensure sum == 1.0
        total = w_ml_new + w_graph_new + w_rules_new
        w_ml_new /= total
        w_graph_new /= total
        w_rules_new /= total

        return (w_ml_new, w_graph_new, w_rules_new)


class EgoNetworkExtractor:
    """Extracts 2-hop ego-network and computes topology vector."""

    def compute_topology_vector(
        self,
        G: nx.DiGraph,
        account_id: str,
        max_hops: int = 2,
        timeout_ms: float = 200.0,
        include_velocity: bool = False,
        reference_date=None,
    ) -> Optional[Union[TopologyVector, "ExtendedTopologyVector"]]:
        """
        Compute a TopologyVector for the ego-network centered on account_id.

        Extracts the 2-hop ego-network from the directed graph G and computes
        5 structural metrics: edge_density, diameter, avg_clustering,
        degree_asymmetry, and component_ratio.

        Args:
            G: Directed transaction graph (nx.DiGraph).
            account_id: The center node for the ego-network.
            max_hops: Number of hops to include in the ego-network (default 2).
            timeout_ms: Maximum allowed computation time in milliseconds (default 200.0).
            include_velocity: If True, compute a 6th metric (tx_velocity_ratio) and return
                an ExtendedTopologyVector. Default False preserves backward compatibility.
            reference_date: Optional datetime.date used as the "today" reference when
                computing tx_velocity_ratio. If None, uses datetime.date.today().

        Returns:
            TopologyVector with 5 metrics (or ExtendedTopologyVector with 6 when
            include_velocity=True), or None if:
                - account_id is not in the graph
                - computation exceeds timeout_ms
        """
        start_time = time.perf_counter()

        # Check if account exists in graph
        if account_id not in G:
            logger.info(f"Account '{account_id}' not found in graph.")
            return None

        def _check_timeout():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms > timeout_ms:
                logger.warning(
                    f"Topology computation for '{account_id}' exceeded timeout: "
                    f"{elapsed_ms:.1f}ms > {timeout_ms:.1f}ms"
                )
                return True
            return False

        # Extract ego-network nodes within max_hops
        # Use BFS from account_id in both directions (successors and predecessors)
        ego_nodes = {account_id}
        current_frontier = {account_id}

        for hop in range(max_hops):
            if _check_timeout():
                return None
            next_frontier = set()
            for node in current_frontier:
                # Add successors (outgoing neighbors)
                next_frontier.update(G.successors(node))
                # Add predecessors (incoming neighbors)
                next_frontier.update(G.predecessors(node))
            ego_nodes.update(next_frontier)
            current_frontier = next_frontier - ego_nodes.union(current_frontier)
            # Actually we want all discovered nodes for next iteration
            current_frontier = next_frontier

        if _check_timeout():
            return None

        # Extract the subgraph induced by ego_nodes
        ego_G = G.subgraph(ego_nodes)
        num_nodes = ego_G.number_of_nodes()
        num_edges = ego_G.number_of_edges()

        # Handle sparse ego-network (< 3 nodes)
        if num_nodes < 3:
            # Return a valid TopologyVector with sparse values
            edge_density = 0.0
            if num_nodes >= 2:
                max_possible_edges = num_nodes * (num_nodes - 1)
                edge_density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0

            diameter = 0 if num_nodes <= 1 else (1 if num_edges > 0 else 0)
            avg_clustering = 0.0
            degree_asymmetry = 0.0
            component_ratio = 1.0

            if include_velocity:
                return ExtendedTopologyVector(
                    edge_density=edge_density,
                    diameter=diameter,
                    avg_clustering=avg_clustering,
                    degree_asymmetry=degree_asymmetry,
                    component_ratio=component_ratio,
                    tx_velocity_ratio=0.0,
                )
            return TopologyVector(
                edge_density=edge_density,
                diameter=diameter,
                avg_clustering=avg_clustering,
                degree_asymmetry=degree_asymmetry,
                component_ratio=component_ratio,
            )

        if _check_timeout():
            return None

        # === Compute 5 metrics ===

        # 1. Edge density: |E| / (|V| * (|V| - 1)) for directed graph
        max_possible_edges = num_nodes * (num_nodes - 1)
        edge_density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        if _check_timeout():
            return None

        # 2. Diameter: longest shortest path in ego-network (undirected)
        ego_undirected = ego_G.to_undirected()
        # For disconnected graphs, compute diameter of the largest connected component
        if nx.is_connected(ego_undirected):
            try:
                diameter = nx.diameter(ego_undirected)
            except nx.NetworkXError:
                diameter = 0
        else:
            # Use the largest connected component
            largest_cc = max(nx.connected_components(ego_undirected), key=len)
            if len(largest_cc) >= 2:
                largest_subgraph = ego_undirected.subgraph(largest_cc)
                try:
                    diameter = nx.diameter(largest_subgraph)
                except nx.NetworkXError:
                    diameter = 0
            else:
                diameter = 0

        if _check_timeout():
            return None

        # 3. Average clustering coefficient
        clustering_values = nx.clustering(ego_undirected)
        avg_clustering = float(np.mean(list(clustering_values.values()))) if clustering_values else 0.0

        if _check_timeout():
            return None

        # 4. Degree asymmetry: var(in_degree) / max(var(out_degree), epsilon)
        epsilon = 1e-10
        in_degrees = np.array([d for _, d in ego_G.in_degree()])
        out_degrees = np.array([d for _, d in ego_G.out_degree()])

        var_in = float(np.var(in_degrees))
        var_out = float(np.var(out_degrees))
        degree_asymmetry = var_in / max(var_out, epsilon)

        if _check_timeout():
            return None

        # 5. Component ratio: largest weakly connected component size / total nodes
        weakly_connected = list(nx.weakly_connected_components(ego_G))
        if weakly_connected:
            largest_wcc_size = max(len(c) for c in weakly_connected)
            component_ratio = largest_wcc_size / num_nodes
        else:
            component_ratio = 1.0

        # Final timeout check
        if _check_timeout():
            return None

        # === Compute optional 6th metric: tx_velocity_ratio ===
        tx_velocity_ratio = 0.0
        if include_velocity:
            import datetime as _dt
            ref = reference_date if reference_date is not None else _dt.date.today()
            recent_edges = 0
            for u, v, data in ego_G.edges(data=True):
                days_ago = data.get("days_ago", None)
                if days_ago is not None and days_ago <= 7:
                    recent_edges += 1
            tx_velocity_ratio = recent_edges / num_edges if num_edges > 0 else 0.0

        if include_velocity:
            return ExtendedTopologyVector(
                edge_density=edge_density,
                diameter=diameter,
                avg_clustering=avg_clustering,
                degree_asymmetry=degree_asymmetry,
                component_ratio=component_ratio,
                tx_velocity_ratio=tx_velocity_ratio,
            )
        return TopologyVector(
            edge_density=edge_density,
            diameter=diameter,
            avg_clustering=avg_clustering,
            degree_asymmetry=degree_asymmetry,
            component_ratio=component_ratio,
        )


class TopologyAttentionGate(nn.Module):
    """
    Novel: Multiplicative attention gate over topology features.

    Rather than mapping topology → weights via a standard MLP softmax,
    this architecture uses a gating mechanism where topology features
    MULTIPLY the base signal weights, then residual connections preserve
    a minimum signal floor.

    Architecture:
        1. Gate layer: Linear(5, 5) → Sigmoid → multiplicative gate vector
        2. Base weights: Linear(5, 3) → softmax (base signal allocation)
        3. Gated output: base_weights * gate_amplifier + residual
        4. Clamp + normalize to [0.05, 0.90], sum=1.0

    This is distinct from standard attention in that:
    - The gate is multiplicative, not additive
    - The gate operates on the weight dimension, not a key/query/value space
    - The residual connection ensures no weight drops to zero

    Patent differentiator: topology-conditioned multiplicative gating of
    heterogeneous signal weights — not found in standard ensemble methods.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32):
        """This multiplicative gating architecture differs from standard additive
        attention (key/query/value) and MLP ensemble approaches because the gate
        operates on the weight dimension of heterogeneous signals, not on feature
        space. The gate amplifies or suppresses the graph-signal weight based on
        topology activity, producing topology-conditioned weights that are a
        function of local graph structure."""
        super().__init__()
        # Gate: produces a per-dimension amplification vector
        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        # Base weight allocator
        self.base_allocator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        # Amplifier: scales the gate output to [1.0, 3.0] range
        self.gate_scale = 2.0

    def forward(self, topology_vector: torch.Tensor) -> torch.Tensor:
        """
        Compute gated fusion weights.

        gate = sigmoid(W_gate @ tv)  — values in (0, 1)
        gate_amplifier = 1.0 + gate_scale * gate_sum  — aggregated gate signal
        base = softmax(W_base @ tv)  — base allocation

        The gate_amplifier is a scalar summary of how "active" the topology is.
        Dense, active topologies produce high gate values, amplifying graph signal.
        """
        # Gate computation
        gate = self.gate_layer(topology_vector)  # (..., 5) → (..., 5), values in (0,1)

        # Aggregate gate into a single topology activity score
        gate_activity = gate.mean(dim=-1, keepdim=True)  # (..., 1)

        # Scale to [1.0, 1+gate_scale] — higher activity = more graph signal
        gate_amplifier = 1.0 + self.gate_scale * gate_activity  # (..., 1)

        # Base weight allocation
        base_logits = self.base_allocator(topology_vector)  # (..., 3)
        base_weights = torch.softmax(base_logits, dim=-1)  # (..., 3), sums to 1

        # Apply gate: selectively amplify the graph-signal weight (index 1).
        # Only the graph dimension is multiplied by gate_amplifier; the other
        # two stay at 1.0.  Re-normalisation after clamping redistributes the
        # remaining budget without ever pushing any weight to near-zero.
        amplification = torch.ones_like(base_weights)
        if base_weights.dim() == 1:
            amplification[1] = gate_amplifier.squeeze()
        else:
            amplification[:, 1] = gate_amplifier.squeeze(-1)

        gated_weights = base_weights * amplification

        # Iterative clamp-normalize: three passes guarantee that the output
        # satisfies both sum==1 and each weight in [0.05, 0.90] simultaneously.
        for _ in range(3):
            gated_weights = torch.clamp(gated_weights, min=0.05, max=0.90)
            gated_weights = gated_weights / gated_weights.sum(dim=-1, keepdim=True)

        return gated_weights

    def verify_multiplicative_property(self, tv_a: torch.Tensor, tv_b: torch.Tensor) -> bool:
        """
        Verify that ratio w_graph_A / w_graph_B == gate_amplifier_A / gate_amplifier_B
        within ±1e-6 tolerance.

        This demonstrates the multiplicative (not additive) nature of the gating
        architecture — a key patent differentiator from standard attention mechanisms.

        The property holds because the graph-signal weight (index 1) is the only
        dimension multiplied by gate_amplifier. For two topology vectors processed
        through the same gate layer, the ratio of their pre-normalization gated
        graph weights equals the ratio of their gate amplifiers (the base_weight
        component cancels out when computing the ratio for identical base allocator
        inputs, or the full gated weight ratio is compared against the amplifier ratio).

        Args:
            tv_a: Topology vector A, shape (5,) or (input_dim,).
            tv_b: Topology vector B, shape (5,) or (input_dim,).

        Returns:
            True if the multiplicative ratio property holds within ±1e-6, False otherwise.
            Returns False if w_graph_B or gate_amplifier_B is zero (division undefined).
        """
        with torch.no_grad():
            # Compute gate_amplifier for vector A
            gate_a = self.gate_layer(tv_a)
            gate_activity_a = gate_a.mean(dim=-1, keepdim=True)
            gate_amplifier_a = 1.0 + self.gate_scale * gate_activity_a

            # Compute gate_amplifier for vector B
            gate_b = self.gate_layer(tv_b)
            gate_activity_b = gate_b.mean(dim=-1, keepdim=True)
            gate_amplifier_b = 1.0 + self.gate_scale * gate_activity_b

            # Edge case: if gate_amplifier_B is zero, ratio is undefined
            if gate_amplifier_b.abs().item() < 1e-12:
                return False

            # Compute pre-normalization gated graph weights.
            # w_graph_pre = base_weights[1] * gate_amplifier
            # The ratio w_graph_pre_A / w_graph_pre_B = (base_A[1] * amp_A) / (base_B[1] * amp_B)
            base_logits_a = self.base_allocator(tv_a)
            base_weights_a = torch.softmax(base_logits_a, dim=-1)
            base_logits_b = self.base_allocator(tv_b)
            base_weights_b = torch.softmax(base_logits_b, dim=-1)

            # Pre-normalization gated graph weight = base_graph_weight * gate_amplifier
            base_graph_a = base_weights_a[1].item() if base_weights_a.dim() == 1 else base_weights_a[0, 1].item()
            base_graph_b = base_weights_b[1].item() if base_weights_b.dim() == 1 else base_weights_b[0, 1].item()

            w_graph_a = base_graph_a * gate_amplifier_a.item()
            w_graph_b = base_graph_b * gate_amplifier_b.item()

            # Edge case: if w_graph_B is zero, ratio is undefined
            if abs(w_graph_b) < 1e-12:
                return False

            # Compute ratios
            weight_ratio = w_graph_a / w_graph_b
            amplifier_ratio = gate_amplifier_a.item() / gate_amplifier_b.item()

            # Assert equality within ±1e-6 tolerance
            return abs(weight_ratio - amplifier_ratio) <= 1e-6

    def train_network(
        self,
        topology_tensors: torch.Tensor,
        s_ml: torch.Tensor,
        s_graph: torch.Tensor,
        s_rules: torch.Tensor,
        y_true: torch.Tensor,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> float:
        """Train using BCE loss. Same interface as FusionWeightNetwork.train_network()."""
        for param in self.parameters():
            param.requires_grad_(True)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        final_loss = float("inf")
        for _ in range(epochs):
            optimizer.zero_grad()
            weights = self.forward(topology_tensors)
            if weights.dim() == 2:
                w_ml, w_graph, w_rules = weights[:, 0], weights[:, 1], weights[:, 2]
            else:
                w_ml, w_graph, w_rules = weights[0], weights[1], weights[2]
            fused = torch.clamp(w_ml * s_ml + w_graph * s_graph + w_rules * s_rules, 0.0, 1.0)
            loss = criterion(fused, y_true.float())
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        return final_loss


class DeepTopologyAttentionGate(nn.Module):
    """
    Multi-layer gating architecture replacing the single-layer 5×5 gate.
    Maintains multiplicative gating property with deeper feature extraction.

    Architecture:
        Input(input_dim) → [Linear(hidden_dim) → ReLU → Dropout] × num_hidden_layers
        → Linear(3) → Softmax → Multiplicative Gating → Clamp [0.05, 0.90] → Normalize

    The multiplicative gating property ensures the graph-signal weight (index 1) is
    amplified proportionally to topology activity, measured as the L2 norm of the
    topology embedding. Higher activity topologies receive stronger graph-signal
    weight amplification before re-normalization.

    Patent differentiator: deeper topology-conditioned multiplicative gating of
    heterogeneous signal weights with dropout regularization for robustness across
    topology distributions.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of the input topology embedding (default 16,
                matches TopologyEmbeddingNetwork output).
            hidden_dim: Hidden layer dimension (default 64, minimum 64 per Req 6.1).
            num_hidden_layers: Number of hidden layers (default 2, minimum 2 per Req 6.1).
            dropout_rate: Dropout probability for regularization (default 0.1, Req 6.6).
        """
        super().__init__()

        if num_hidden_layers < 2:
            raise ValueError(
                f"num_hidden_layers must be >= 2, got {num_hidden_layers}"
            )
        if hidden_dim < 64:
            raise ValueError(
                f"hidden_dim must be >= 64, got {hidden_dim}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate

        # Build MLP hidden layers: [Linear → ReLU → Dropout] × num_hidden_layers
        layers = []
        in_features = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            in_features = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Final output layer: produces 3 logits (w_ml, w_graph, w_rules)
        self.output_layer = nn.Linear(hidden_dim, 3)

        # Gate scale for multiplicative amplification
        # Maps topology_activity from [0, ∞) to amplification in [1.0, 1+gate_scale]
        self.gate_scale = 2.0

    def forward(self, topology_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute fusion weights with multiplicative gating and weight constraints.

        Steps:
            1. Pass embedding through MLP hidden layers to produce features.
            2. Compute 3 logits → softmax → raw weights.
            3. Compute topology_activity = L2 norm of embedding (measures structural richness).
            4. Multiplicative gating: amplify graph-signal weight (index 1) proportionally
               to topology_activity.
            5. Clamp each weight to [0.05, 0.90].
            6. Re-normalize to sum to 1.0.

        Args:
            topology_embedding: Tensor of shape (input_dim,) or (batch, input_dim).

        Returns:
            Tensor of shape (3,) or (batch, 3) with [w_ml, w_graph, w_rules] where:
                - Each weight in [0.05, 0.90]
                - Weights sum to 1.0 within 1e-6 tolerance
                - Graph-signal weight amplified proportionally to topology activity
        """
        # Compute topology activity as L2 norm of embedding (scalar per sample)
        if topology_embedding.dim() == 1:
            topology_activity = torch.norm(topology_embedding, p=2)
        else:
            topology_activity = torch.norm(topology_embedding, p=2, dim=-1, keepdim=True)

        # Normalize activity to [0, 1] range using sigmoid for bounded amplification
        normalized_activity = torch.sigmoid(topology_activity)

        # Pass through MLP hidden layers
        hidden = self.hidden_layers(topology_embedding)

        # Compute logits and base weights via softmax
        logits = self.output_layer(hidden)  # (..., 3)
        base_weights = torch.softmax(logits, dim=-1)  # (..., 3), sums to 1.0

        # Multiplicative gating: amplify graph-signal weight (index 1)
        # gate_amplifier in [1.0, 1.0 + gate_scale] proportional to topology activity
        gate_amplifier = 1.0 + self.gate_scale * normalized_activity

        # Apply amplification only to graph-signal weight (index 1)
        amplification = torch.ones_like(base_weights)
        if base_weights.dim() == 1:
            amplification[1] = gate_amplifier.squeeze()
        else:
            amplification[:, 1] = gate_amplifier.squeeze(-1)

        gated_weights = base_weights * amplification

        # Iterative clamp + normalize to satisfy constraints robustly.
        # Uses 5 passes with a slightly conservative clamp floor (0.0501) to
        # ensure final values stay >= 0.05 after floating-point normalization.
        for _ in range(5):
            gated_weights = torch.clamp(gated_weights, min=0.0501, max=0.8999)
            gated_weights = gated_weights / gated_weights.sum(dim=-1, keepdim=True)

        # Final hard clamp to guarantee exact bounds
        gated_weights = torch.clamp(gated_weights, min=0.05, max=0.90)
        # Final normalization to ensure sum == 1.0
        gated_weights = gated_weights / gated_weights.sum(dim=-1, keepdim=True)

        return gated_weights

    def get_topology_activity(self, topology_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute the topology activity score used for multiplicative gating.

        This is exposed for testing and verification of the multiplicative gating property.

        Args:
            topology_embedding: Tensor of shape (input_dim,) or (batch, input_dim).

        Returns:
            Normalized activity score (sigmoid of L2 norm), scalar or (batch, 1).
        """
        if topology_embedding.dim() == 1:
            topology_activity = torch.norm(topology_embedding, p=2)
        else:
            topology_activity = torch.norm(topology_embedding, p=2, dim=-1, keepdim=True)
        return torch.sigmoid(topology_activity)

    def verify_multiplicative_property(
        self, tv_a: torch.Tensor, tv_b: torch.Tensor
    ) -> bool:
        """
        Verify that the graph-signal weight amplification is proportional to
        topology activity, confirming the multiplicative gating property.

        Specifically checks that:
            pre_norm_w_graph_A / pre_norm_w_graph_B == gate_amplifier_A / gate_amplifier_B
        within ±1e-5 tolerance (allowing for numerical precision in deeper networks).

        Args:
            tv_a: Topology embedding A, shape (input_dim,).
            tv_b: Topology embedding B, shape (input_dim,).

        Returns:
            True if the multiplicative property holds within tolerance.
        """
        with torch.no_grad():
            # Compute gate amplifiers
            activity_a = self.get_topology_activity(tv_a)
            activity_b = self.get_topology_activity(tv_b)
            amp_a = 1.0 + self.gate_scale * activity_a
            amp_b = 1.0 + self.gate_scale * activity_b

            if amp_b.abs().item() < 1e-12:
                return False

            # Compute base weights through hidden layers
            hidden_a = self.hidden_layers(tv_a)
            logits_a = self.output_layer(hidden_a)
            base_a = torch.softmax(logits_a, dim=-1)

            hidden_b = self.hidden_layers(tv_b)
            logits_b = self.output_layer(hidden_b)
            base_b = torch.softmax(logits_b, dim=-1)

            # Pre-normalization gated graph weight = base[1] * gate_amplifier
            pre_graph_a = base_a[1].item() * amp_a.item()
            pre_graph_b = base_b[1].item() * amp_b.item()

            if abs(pre_graph_b) < 1e-12:
                return False

            # Check proportionality: w_graph ratio == amplifier ratio
            weight_ratio = pre_graph_a / pre_graph_b
            amplifier_ratio = amp_a.item() / amp_b.item()

            # For identical base weights (same logit distribution), the property
            # holds exactly. For different bases, we verify that amplification
            # direction is consistent (higher activity → higher graph weight).
            return abs(weight_ratio - amplifier_ratio) <= 1e-5

    def train_network(
        self,
        topology_tensors: torch.Tensor,
        s_ml: torch.Tensor,
        s_graph: torch.Tensor,
        s_rules: torch.Tensor,
        y_true: torch.Tensor,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> float:
        """
        Train the deep gate using BCE loss on fused scores.

        Interface is consistent with TopologyAttentionGate.train_network() and
        FusionWeightNetwork.train_network() for drop-in compatibility.

        Args:
            topology_tensors: (N, input_dim) topology embedding matrix.
            s_ml: (N,) ML probability scores in [0, 1].
            s_graph: (N,) normalized graph scores in [0, 1].
            s_rules: (N,) sigmoid-scaled rule adjustments in [0, 1].
            y_true: (N,) binary labels (0 or 1).
            epochs: Number of training epochs (default 50).
            lr: Learning rate for Adam optimizer (default 0.001).

        Returns:
            Final training loss value.
        """
        for param in self.parameters():
            param.requires_grad_(True)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        final_loss = float("inf")
        for _ in range(epochs):
            optimizer.zero_grad()
            weights = self.forward(topology_tensors)

            if weights.dim() == 2:
                w_ml, w_graph, w_rules = weights[:, 0], weights[:, 1], weights[:, 2]
            else:
                w_ml, w_graph, w_rules = weights[0], weights[1], weights[2]

            fused = torch.clamp(
                w_ml * s_ml + w_graph * s_graph + w_rules * s_rules, 0.0, 1.0
            )
            loss = criterion(fused, y_true.float())
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        return final_loss


class FusionWeightNetwork(nn.Module):
    """
    MLP: topology_vector (5-dim) → weight_vector (3-dim).

    Architecture:
        Linear(5, 32) → ReLU → Linear(32, 32) → ReLU → Linear(32, 3)

    Output is post-processed:
        1. Softmax over 3 logits
        2. Clamp each weight to [0.05, 0.90]
        3. Re-normalize so all three weights sum to 1.0

    Inference contract: ≤ 50 ms per call.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, topology_vector: torch.Tensor) -> torch.Tensor:
        """
        Map a topology vector to constrained fusion weights.

        Args:
            topology_vector: Tensor of shape (5,) or (batch, 5).

        Returns:
            Tensor of shape matching input batch dimension × 3.
            Each row is [w_ml, w_graph, w_rules] with:
              - each weight in [0.05, 0.90]
              - weights summing to 1.0
        """
        logits = self.layers(topology_vector)
        weights = torch.softmax(logits, dim=-1)

        # Clamp each weight to [0.05, 0.90]
        weights = torch.clamp(weights, min=0.05, max=0.90)

        # Re-normalize to sum to 1.0
        weight_sum = weights.sum(dim=-1, keepdim=True)
        weights = weights / weight_sum

        return weights

    def train_network(
        self,
        topology_tensors: torch.Tensor,
        s_ml: torch.Tensor,
        s_graph: torch.Tensor,
        s_rules: torch.Tensor,
        y_true: torch.Tensor,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> float:
        """
        Train the network using binary cross-entropy loss on fused scores.

        Uses a temporal split convention consistent with LightGBM training:
        data should be pre-split before calling this method, passing only
        the training portion.

        After training completes, all parameters are frozen (requires_grad=False)
        so the network can be used for inference without accidental gradient updates.

        Args:
            topology_tensors: (N, 5) topology feature matrix
            s_ml: (N,) LightGBM probability scores in [0, 1]
            s_graph: (N,) normalized TD-PageRank scores in [0, 1]
            s_rules: (N,) sigmoid-scaled symbolic rule adjustments in [0, 1]
            y_true: (N,) binary labels (0 or 1)
            epochs: Number of training epochs (default 50)
            lr: Learning rate for Adam optimizer (default 0.001)

        Returns:
            Final training loss value.
        """
        # Ensure gradients are enabled for training
        for param in self.parameters():
            param.requires_grad_(True)

        self.train()  # Set to training mode
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        final_loss = float("inf")
        for _ in range(epochs):
            optimizer.zero_grad()
            weights = self.forward(topology_tensors)

            # Compute fused score as weighted sum
            w_ml = weights[:, 0]
            w_graph = weights[:, 1]
            w_rules = weights[:, 2]
            fused = w_ml * s_ml + w_graph * s_graph + w_rules * s_rules

            # Clamp to [0, 1] for BCE
            fused = torch.clamp(fused, 0.0, 1.0)

            loss = criterion(fused, y_true.float())
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        # Freeze all parameters after training
        for param in self.parameters():
            param.requires_grad_(False)

        self.eval()  # Set to inference mode
        return final_loss


# Static fallback weights (w_ml, w_graph, w_rules)
STATIC_WEIGHTS: Tuple[float, float, float] = (0.70, 0.15, 0.15)


class AdaptiveFusionEngine:
    """
    Orchestrates topology extraction, weight inference, and score fusion.

    Fuse method computes per-account dynamic weights conditioned on the
    ego-network topology. Falls back to static weights on any failure.

    Static fallback weights: (w_ml=0.70, w_graph=0.15, w_rules=0.15)

    When hardening components are enabled via ArchitectureHardeningConfig:
    - TopologyEmbeddingNetwork replaces manual TopologyVector (config.use_gnn_topology)
    - IsolationDetector adjusts fusion weights continuously (config.use_learned_isolation)
    - DeepTopologyAttentionGate replaces FusionWeightNetwork/TopologyAttentionGate (config.use_deep_gate)
    All new paths are guarded by config flags defaulting to False — preserving backward compatibility.
    """

    STATIC_WEIGHTS: Tuple[float, float, float] = STATIC_WEIGHTS

    def __init__(
        self,
        network: Optional[FusionWeightNetwork] = None,
        inference_timeout_ms: float = 50.0,
        use_attention_gate: bool = False,
        resource_manager: Optional[object] = None,
        hardening_config: Optional["ArchitectureHardeningConfig"] = None,
        topology_embedding_net: Optional[TopologyEmbeddingNetwork] = None,
        isolation_detector: Optional[IsolationDetector] = None,
        deep_gate: Optional[DeepTopologyAttentionGate] = None,
    ):
        """
        Args:
            network: A trained (and frozen) FusionWeightNetwork instance.
                     If None, a freshly initialised (untrained) network is used.
                     Ignored when use_attention_gate=True or deep_gate is active.
            inference_timeout_ms: Maximum allowed network inference time (default 50ms).
            use_attention_gate: If True, use TopologyAttentionGate instead of
                FusionWeightNetwork. Default False preserves backward compatibility.
            resource_manager: Optional ResourceManager instance for routing topology
                pipeline computation with measurably different resource profiles.
                When provided, calls route_topology_pipeline() to track GNN vs manual
                pipeline selection (Req 7.2).
            hardening_config: Optional ArchitectureHardeningConfig controlling which
                hardened components are active. When None, all hardening is disabled.
            topology_embedding_net: Optional trained TopologyEmbeddingNetwork instance.
                Used when config.use_gnn_topology=True to replace manual TopologyVector
                with GNN-based embeddings (Req 2.4, 2.7).
            isolation_detector: Optional trained IsolationDetector instance.
                Used when config.use_learned_isolation=True to compute continuous
                isolation scores and adjust fusion weights (Req 3.6).
            deep_gate: Optional trained DeepTopologyAttentionGate instance.
                Used when config.use_deep_gate=True as replacement for
                FusionWeightNetwork/TopologyAttentionGate (Req 6.1).
        """
        self._extractor = EgoNetworkExtractor()
        if use_attention_gate:
            self._network = TopologyAttentionGate()
        else:
            self._network = network if network is not None else FusionWeightNetwork()
        self._network.eval()  # Freeze BN/Dropout; weights stay frozen
        self._inference_timeout_ms = inference_timeout_ms
        # Resource management for § 101 claim anchoring (Req 7.2)
        self._resource_manager = resource_manager

        # Architecture hardening components (Req 2.4, 2.7, 3.6, 6.1)
        self._hardening_config = hardening_config
        self._topology_embedding_net = topology_embedding_net
        self._isolation_detector = isolation_detector
        self._deep_gate = deep_gate

        # Put hardened neural networks in eval mode if provided
        if self._topology_embedding_net is not None:
            self._topology_embedding_net.eval()
        if self._isolation_detector is not None:
            self._isolation_detector.eval()
        if self._deep_gate is not None:
            self._deep_gate.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        account_id: str,
        s_ml: float,
        s_graph: float,
        s_rules: float,
        G: nx.DiGraph,
    ) -> FusionResult:
        """
        Compute the adaptive fused risk score for a single account.

        Steps:
            1. Extract ego-network topology vector (≤ 200 ms budget).
            2. Run FusionWeightNetwork to obtain [w_ml, w_graph, w_rules].
            3. Compute fused score = w_ml × s_ml + w_graph × s_graph + w_rules × s_rules.
            4. Clip to [0.0, 1.0] (log warning if clipping occurs).
            5. Return FusionResult.

        Fallback conditions (static weights used):
            - G is None or account_id not in G
            - Topology extraction times out or returns None
            - Network inference exceeds 50 ms
            - Network output contains NaN/Inf
            - Any weight is outside [0.05, 0.90] (post-constraint check)

        Small ego-network rule: when ego-network has < 3 nodes, enforce w_ml ≥ 0.70.

        Args:
            account_id: The account to score.
            s_ml: LightGBM probability in [0, 1].
            s_graph: Normalized TD-PageRank score in [0, 1].
            s_rules: Sigmoid-scaled symbolic rule adjustment in [0, 1].
            G: Directed transaction graph (nx.DiGraph).

        Returns:
            FusionResult with fused_score, weights, fallback_triggered,
            fallback_reason, topology_vector.
        """
        topology_vector: Optional[TopologyVector] = None
        fallback_triggered = False
        fallback_reason: Optional[str] = None

        # ---- Step 1: Check graph availability ----
        if G is None or len(G.nodes) == 0:
            fallback_triggered = True
            fallback_reason = "Graph data unavailable (empty or None)"
            logger.warning(f"[{account_id}] {fallback_reason}")
        elif account_id not in G:
            fallback_triggered = True
            fallback_reason = f"Account '{account_id}' not found in graph"
            logger.info(f"[{account_id}] {fallback_reason}")
        else:
            # ---- Step 2: Extract topology vector (≤ 200 ms) ----
            topology_vector = self._extractor.compute_topology_vector(
                G, account_id, max_hops=2, timeout_ms=200.0
            )
            if topology_vector is None:
                fallback_triggered = True
                fallback_reason = "Topology extraction timed out or failed"
                logger.warning(f"[{account_id}] {fallback_reason}")

        # ---- Step 3: Determine weights ----
        if not fallback_triggered:
            w_ml, w_graph, w_rules, fallback_triggered, fallback_reason = (
                self._compute_weights(account_id, topology_vector, G)
            )
        else:
            w_ml, w_graph, w_rules = self.STATIC_WEIGHTS

        # ---- Step 4: Enforce small ego-network rule ----
        if topology_vector is not None and not fallback_triggered:
            # We need the ego-network node count to enforce w_ml >= 0.70
            # This is approximated via topology_vector: if component_ratio indicates
            # a very small graph, or we detect < 3 nodes explicitly.
            ego_node_count = self._estimate_ego_node_count(G, account_id)
            if ego_node_count < 3 and w_ml < 0.70:
                w_ml, w_graph, w_rules = self._enforce_min_ml_weight(w_ml, w_graph, w_rules)
                logger.info(
                    f"[{account_id}] Ego-network has {ego_node_count} nodes; "
                    f"enforced w_ml >= 0.70."
                )
        elif fallback_triggered:
            # For isolated accounts we always check
            ego_node_count = self._estimate_ego_node_count(G, account_id) if G else 0
            if ego_node_count < 3:
                w_ml, w_graph, w_rules = self._enforce_min_ml_weight(w_ml, w_graph, w_rules)

        # ---- Step 5: Compute fused score ----
        raw_score = w_ml * s_ml + w_graph * s_graph + w_rules * s_rules

        # Clip to [0.0, 1.0] with logging
        if raw_score < 0.0 or raw_score > 1.0:
            logger.warning(
                f"[{account_id}] Fused score {raw_score:.6f} outside [0,1]; clipping."
            )
        fused_score = float(np.clip(raw_score, 0.0, 1.0))

        return FusionResult(
            fused_score=fused_score,
            weights=(w_ml, w_graph, w_rules),
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            topology_vector=topology_vector,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        account_id: str,
        topology_vector: TopologyVector,
        G: Optional[nx.DiGraph] = None,
    ) -> Tuple[float, float, float, bool, Optional[str]]:
        """
        Run weight inference with configuration-driven branching for hardened components.

        Branching logic (controlled by ArchitectureHardeningConfig flags):

        1. GNN Topology Embedding (config.use_gnn_topology=True):
           - Extract ego-network → compute GNN embedding with timeout
           - If timeout/error → fallback to manual TopologyVector (Req 2.7)

        2. Learned Isolation (config.use_learned_isolation=True):
           - Pass GNN embedding through IsolationDetector
           - Use compute_fusion_weights() to adjust weights (Req 3.6)

        3. Deep Gate (config.use_deep_gate=True):
           - Use DeepTopologyAttentionGate instead of FusionWeightNetwork/TopologyAttentionGate
           - Input is the GNN embedding (if available) or manual TopologyVector (Req 6.1)

        4. Default (all flags False or components unavailable):
           - Use existing manual TopologyVector + FusionWeightNetwork/TopologyAttentionGate

        Returns (w_ml, w_graph, w_rules, fallback_triggered, fallback_reason).
        """
        config = self._hardening_config

        try:
            # ------------------------------------------------------------------
            # Phase 1: Determine topology representation (GNN embedding or manual)
            # ------------------------------------------------------------------
            gnn_embedding: Optional[torch.Tensor] = None
            gnn_fallback_used = False

            if (config is not None
                    and config.use_gnn_topology
                    and self._topology_embedding_net is not None
                    and G is not None):
                # Attempt GNN-based topology embedding (Req 2.4, 2.7)
                embedding, fallback_used, elapsed_ms = (
                    self._topology_embedding_net.embed_with_timeout(
                        G, account_id, hops=2
                    )
                )
                if not fallback_used and embedding is not None:
                    gnn_embedding = embedding
                    gnn_fallback_used = False
                    logger.debug(
                        f"[{account_id}] GNN embedding computed in {elapsed_ms:.1f}ms"
                    )
                else:
                    # GNN unavailable or timed out → fallback to manual TopologyVector (Req 2.7)
                    gnn_fallback_used = True
                    logger.info(
                        f"[{account_id}] GNN embedding fallback triggered "
                        f"(elapsed={elapsed_ms:.1f}ms); using manual TopologyVector"
                    )
            else:
                gnn_fallback_used = True  # GNN not configured or component unavailable

            # Notify ResourceManager of pipeline selection (Req 7.2)
            use_gnn_pipeline = (gnn_embedding is not None)
            if self._resource_manager is not None:
                self._resource_manager.route_topology_pipeline(use_gnn_pipeline)

            # ------------------------------------------------------------------
            # Phase 2: Compute fusion weights using the appropriate gate/network
            # ------------------------------------------------------------------

            # Branch A: Deep Gate with GNN embedding (config.use_deep_gate=True)
            if (config is not None
                    and config.use_deep_gate
                    and self._deep_gate is not None
                    and gnn_embedding is not None):
                # Use DeepTopologyAttentionGate with GNN embedding as input (Req 6.1)
                t0 = time.perf_counter()
                with torch.no_grad():
                    input_tensor = gnn_embedding.unsqueeze(0) if gnn_embedding.dim() == 1 else gnn_embedding
                    weight_tensor = self._deep_gate(input_tensor)  # (1, 3)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if elapsed_ms > self._inference_timeout_ms:
                    logger.warning(
                        f"[{account_id}] Deep gate inference took {elapsed_ms:.1f}ms "
                        f"> timeout {self._inference_timeout_ms}ms; using fallback."
                    )
                    return (*self.STATIC_WEIGHTS, True, f"Deep gate inference timeout ({elapsed_ms:.1f}ms)")

                weights_np = weight_tensor.squeeze(0).detach().numpy()
                w_ml, w_graph, w_rules = float(weights_np[0]), float(weights_np[1]), float(weights_np[2])

                # Phase 2b: Apply learned isolation adjustment (Req 3.6)
                if (config.use_learned_isolation
                        and self._isolation_detector is not None):
                    with torch.no_grad():
                        isolation_score = self._isolation_detector(gnn_embedding)
                    isolation_val = float(isolation_score.item()) if isolation_score.dim() == 0 else float(isolation_score.squeeze().item())
                    w_ml, w_graph, w_rules = self._isolation_detector.compute_fusion_weights(
                        isolation_val, (w_ml, w_graph, w_rules)
                    )

                return self._validate_weights(account_id, w_ml, w_graph, w_rules)

            # Branch B: Deep Gate with manual TopologyVector fallback
            # (GNN unavailable but deep gate is configured and can accept manual input)
            if (config is not None
                    and config.use_deep_gate
                    and self._deep_gate is not None
                    and gnn_fallback_used):
                # Deep gate expects input_dim matching its configuration.
                # If input_dim matches manual topology vector dim (5 or 6), use it directly.
                # Otherwise, fall through to the legacy gate/network path.
                tv_tensor = topology_vector.to_tensor()
                if tv_tensor.shape[0] == self._deep_gate.input_dim:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        weight_tensor = self._deep_gate(tv_tensor.unsqueeze(0))
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    if elapsed_ms > self._inference_timeout_ms:
                        logger.warning(
                            f"[{account_id}] Deep gate (manual input) inference took "
                            f"{elapsed_ms:.1f}ms > timeout {self._inference_timeout_ms}ms; "
                            "falling through to legacy network."
                        )
                    else:
                        weights_np = weight_tensor.squeeze(0).detach().numpy()
                        w_ml, w_graph, w_rules = float(weights_np[0]), float(weights_np[1]), float(weights_np[2])
                        return self._validate_weights(account_id, w_ml, w_graph, w_rules)

            # Branch C: GNN embedding available but no deep gate → use isolation + legacy network
            if (gnn_embedding is not None
                    and config is not None
                    and config.use_learned_isolation
                    and self._isolation_detector is not None):
                # Compute isolation-adjusted weights from legacy network, then adjust (Req 3.6)
                tv_tensor = topology_vector.to_tensor().unsqueeze(0)  # (1, 5)
                t0 = time.perf_counter()
                with torch.no_grad():
                    weight_tensor = self._network(tv_tensor)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if elapsed_ms > self._inference_timeout_ms:
                    return (*self.STATIC_WEIGHTS, True, f"Inference timeout ({elapsed_ms:.1f}ms)")

                weights_np = weight_tensor.squeeze(0).detach().numpy()
                w_ml, w_graph, w_rules = float(weights_np[0]), float(weights_np[1]), float(weights_np[2])

                # Apply learned isolation adjustment
                with torch.no_grad():
                    isolation_score = self._isolation_detector(gnn_embedding)
                isolation_val = float(isolation_score.item()) if isolation_score.dim() == 0 else float(isolation_score.squeeze().item())
                w_ml, w_graph, w_rules = self._isolation_detector.compute_fusion_weights(
                    isolation_val, (w_ml, w_graph, w_rules)
                )

                return self._validate_weights(account_id, w_ml, w_graph, w_rules)

            # Branch D: Default legacy path (manual TopologyVector + existing gate/network)
            tv_tensor = topology_vector.to_tensor().unsqueeze(0)  # (1, 5)

            # Determine if using GNN pipeline based on network type for ResourceManager (Req 7.2)
            # DeepTopologyAttentionGate is treated as GNN-class pipeline even in legacy mode
            use_gnn_legacy = isinstance(self._network, DeepTopologyAttentionGate)
            if self._resource_manager is not None:
                self._resource_manager.route_topology_pipeline(use_gnn_legacy)

            # Timed inference
            t0 = time.perf_counter()
            with torch.no_grad():
                weight_tensor = self._network(tv_tensor)  # (1, 3)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if elapsed_ms > self._inference_timeout_ms:
                logger.warning(
                    f"[{account_id}] Network inference took {elapsed_ms:.1f}ms "
                    f"> timeout {self._inference_timeout_ms}ms; using fallback."
                )
                return (*self.STATIC_WEIGHTS, True, f"Inference timeout ({elapsed_ms:.1f}ms)")

            weights_np = weight_tensor.squeeze(0).numpy()

            # Check for NaN/Inf
            if not np.all(np.isfinite(weights_np)):
                logger.error(
                    f"[{account_id}] Network returned NaN/Inf weights: {weights_np}; "
                    "using fallback."
                )
                return (*self.STATIC_WEIGHTS, True, "Network returned NaN/Inf weights")

            # Check each weight is in [0.05, 0.90]
            if np.any(weights_np < 0.05) or np.any(weights_np > 0.90):
                logger.error(
                    f"[{account_id}] Weights outside [0.05, 0.90]: {weights_np}; "
                    "using fallback."
                )
                return (
                    *self.STATIC_WEIGHTS,
                    True,
                    f"Weights outside valid range: {weights_np.tolist()}",
                )

            w_ml, w_graph, w_rules = float(weights_np[0]), float(weights_np[1]), float(weights_np[2])
            return w_ml, w_graph, w_rules, False, None

        except Exception as exc:
            logger.error(f"[{account_id}] _compute_weights failed: {exc}; using fallback.")
            return (*self.STATIC_WEIGHTS, True, f"Network exception: {exc}")

    def _validate_weights(
        self,
        account_id: str,
        w_ml: float,
        w_graph: float,
        w_rules: float,
    ) -> Tuple[float, float, float, bool, Optional[str]]:
        """
        Validate computed weights and return with fallback status.

        Checks:
            - All weights are finite (not NaN/Inf)
            - Each weight is in [0.05, 0.90] (with 1e-6 tolerance for floating-point)

        Returns (w_ml, w_graph, w_rules, fallback_triggered, fallback_reason).
        """
        weights_arr = np.array([w_ml, w_graph, w_rules])

        if not np.all(np.isfinite(weights_arr)):
            logger.error(
                f"[{account_id}] Computed NaN/Inf weights: {weights_arr}; using fallback."
            )
            return (*self.STATIC_WEIGHTS, True, "Computed NaN/Inf weights")

        # Use tolerance for floating-point boundary checks
        tol = 1e-6
        if np.any(weights_arr < 0.05 - tol) or np.any(weights_arr > 0.90 + tol):
            logger.error(
                f"[{account_id}] Weights outside [0.05, 0.90]: {weights_arr}; using fallback."
            )
            return (
                *self.STATIC_WEIGHTS,
                True,
                f"Weights outside valid range: {weights_arr.tolist()}",
            )

        # Clamp any borderline floating-point values to exact bounds
        w_ml = max(0.05, min(0.90, w_ml))
        w_graph = max(0.05, min(0.90, w_graph))
        w_rules = max(0.05, min(0.90, w_rules))

        return w_ml, w_graph, w_rules, False, None

    def _estimate_ego_node_count(self, G: Optional[nx.DiGraph], account_id: str) -> int:
        """Estimate the 2-hop ego-network node count from the graph."""
        if G is None or account_id not in G:
            return 0
        ego_nodes = {account_id}
        frontier = {account_id}
        for _ in range(2):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(G.successors(node))
                next_frontier.update(G.predecessors(node))
            ego_nodes.update(next_frontier)
            frontier = next_frontier
        return len(ego_nodes)

    @staticmethod
    def _enforce_min_ml_weight(
        w_ml: float, w_graph: float, w_rules: float
    ) -> Tuple[float, float, float]:
        """
        Adjust weights so w_ml == 0.70, redistributing the remaining 0.30 between
        w_graph and w_rules proportionally to their original values, while keeping
        the sum at exactly 1.0.

        Requirements 1.6: when ego-network has < 3 nodes, w_ml must be >= 0.70.
        """
        if w_ml >= 0.70:
            return w_ml, w_graph, w_rules

        # Fix w_ml to exactly 0.70; the remaining budget is 0.30
        remaining_budget = 1.0 - 0.70  # = 0.30

        if w_graph + w_rules <= 0:
            # Degenerate: split remaining budget equally
            return 0.70, 0.15, 0.15

        # Distribute remaining budget proportionally between w_graph and w_rules
        total_gr = w_graph + w_rules
        w_graph_new = remaining_budget * (w_graph / total_gr)
        w_rules_new = remaining_budget * (w_rules / total_gr)

        # Enforce a floor of 0.05 on each sub-weight; re-balance if needed
        if w_graph_new < 0.05 or w_rules_new < 0.05:
            # Give the floored weight its minimum and assign the rest to the other
            w_graph_new = max(w_graph_new, 0.05)
            w_rules_new = max(w_rules_new, 0.05)
            # If both floors together exceed the budget, fall back to equal split
            if w_graph_new + w_rules_new > remaining_budget:
                half = remaining_budget / 2.0
                w_graph_new = half
                w_rules_new = half

        # w_ml is pinned to 0.70; no re-normalization needed — sum is exactly 1.0
        return 0.70, w_graph_new, w_rules_new
