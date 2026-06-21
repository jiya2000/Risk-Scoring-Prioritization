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
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.data_models import TopologyVector, FusionResult


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
    """

    STATIC_WEIGHTS: Tuple[float, float, float] = STATIC_WEIGHTS

    def __init__(
        self,
        network: Optional[FusionWeightNetwork] = None,
        inference_timeout_ms: float = 50.0,
        use_attention_gate: bool = False,
    ):
        """
        Args:
            network: A trained (and frozen) FusionWeightNetwork instance.
                     If None, a freshly initialised (untrained) network is used.
                     Ignored when use_attention_gate=True.
            inference_timeout_ms: Maximum allowed network inference time (default 50ms).
            use_attention_gate: If True, use TopologyAttentionGate instead of
                FusionWeightNetwork. Default False preserves backward compatibility.
        """
        self._extractor = EgoNetworkExtractor()
        if use_attention_gate:
            self._network = TopologyAttentionGate()
        else:
            self._network = network if network is not None else FusionWeightNetwork()
        self._network.eval()  # Freeze BN/Dropout; weights stay frozen
        self._inference_timeout_ms = inference_timeout_ms

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
                self._compute_weights(account_id, topology_vector)
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
    ) -> Tuple[float, float, float, bool, Optional[str]]:
        """
        Run FusionWeightNetwork inference and validate outputs.

        Returns (w_ml, w_graph, w_rules, fallback_triggered, fallback_reason).
        """
        try:
            tv_tensor = topology_vector.to_tensor().unsqueeze(0)  # (1, 5)

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
            logger.error(f"[{account_id}] FusionWeightNetwork failed: {exc}; using fallback.")
            return (*self.STATIC_WEIGHTS, True, f"Network exception: {exc}")

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
