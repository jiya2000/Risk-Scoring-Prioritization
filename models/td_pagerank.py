"""
Temporal-Decay PageRank (TD-PageRank) Engine.

Implements a modified PageRank algorithm with:
- Exponential temporal decay on edge weights based on transaction age
- Cycle penalization for nodes in strongly connected components
- Deterministic results within 1e-12 floating-point tolerance

Replaces standard nx.pagerank() as the primary graph centrality signal.

Novel mechanisms (patent differentiators):
1. Exponential temporal decay with configurable half-life embedded in power iteration
2. Burst-velocity amplification: nodes with sudden transaction surges receive amplified
   centrality — captures smurfing/structuring behaviour (§ 103 differentiator)
3. Log-amount normalization: prevents large single transactions from dominating
   centrality computation (prior art does not address this flaw)
4. Directional SCC penalty: asymmetric dampening based on intra-SCC flow direction —
   collectors (inflow-dominant) are penalised harder than distributors (outflow-dominant)
5. Dormant node suppression: nodes inactive > 30 days capped at 0.1× max score
"""

import sys
import os
from datetime import date
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
import networkx as nx

# Ensure project root is in path for cross-module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.data_models import TDPageRankResult


class TDPageRankEngine:
    """
    Temporal-Decay PageRank with cycle penalization and novel AML differentiators.

    Computes a modified PageRank where:
    - Edge weights decay exponentially with transaction age
    - Nodes in strongly connected components (size > 2) receive a penalty
      when >80% of their incident temporal weight is intra-SCC
    - Burst-velocity amplification amplifies nodes with sudden transaction surges
    - Log-amount normalization prevents large single transactions from dominating
    - Directional SCC penalty applies asymmetric dampening based on flow direction

    Parameters:
        half_life_days: Half-life for exponential decay (default 7.0 days)
        damping: PageRank damping factor (default 0.85)
        cycle_penalty: Multiplicative penalty for cycle nodes (default 0.5)
        max_iter: Maximum power iteration steps (default 100)
        tol: L1 norm convergence threshold (default 1e-6)
        burst_penalty: Unused reserve parameter for future burst-direction weighting
                       (burst mechanism uses burst_velocity_ratio, not a penalty)
        burst_window_days: Lookback window (days) for burst-velocity detection (default 3)
        log_amount_scale: If True, apply log1p normalisation to amounts before
                          computing temporal weights (default False to preserve
                          backward-compatible decay formula; opt-in for novel behaviour)
        asymmetric_scc_penalty: If True, apply directional (asymmetric) SCC penalty
                                 distinguishing collectors vs distributors (default False
                                 to preserve backward-compatible cycle-penalty behaviour)
    """

    def __init__(
        self,
        half_life_days: float = 7.0,
        damping: float = 0.85,
        cycle_penalty: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
        burst_penalty: float = 0.3,
        burst_window_days: int = 3,
        log_amount_scale: bool = False,
        asymmetric_scc_penalty: bool = False,
    ):
        self.half_life_days = half_life_days
        self.damping = damping
        self.cycle_penalty = cycle_penalty
        self.max_iter = max_iter
        self.tol = tol
        self.burst_penalty = burst_penalty           # reserved / future directional weight
        self.burst_window_days = burst_window_days   # lookback window for burst detection
        self.log_amount_scale = log_amount_scale     # log1p normalisation flag
        self.asymmetric_scc_penalty = asymmetric_scc_penalty  # directional SCC penalty flag
        # Decay rate: λ = ln(2) / half_life_days
        self.decay_lambda = 0.693 / half_life_days

    def compute(
        self,
        edges_df: pd.DataFrame,
        reference_date: Optional[date] = None,
    ) -> TDPageRankResult:
        """
        Compute TD-PageRank scores for all nodes in the transaction graph.

        Args:
            edges_df: DataFrame with columns [Sender_account, Receiver_account,
                      amount_local_npr, Date]
            reference_date: Date for computing Edge_Age. Defaults to max(edges_df.Date).

        Returns:
            TDPageRankResult with per-node scores, flags, and metadata.
        """
        # Handle empty input
        if edges_df is None or len(edges_df) == 0:
            return TDPageRankResult(
                scores={},
                normalized_scores={},
                cycle_member={},
                decay_impact={},
                converged=True,
                iterations=0,
                reference_date=reference_date or date.today(),
            )

        # Determine reference date
        if reference_date is None:
            max_date = pd.to_datetime(edges_df["Date"]).max()
            if pd.isna(max_date):
                reference_date = date.today()
            else:
                reference_date = max_date.date() if hasattr(max_date, 'date') else max_date

        # Compute temporal weights
        temporal_weights = self._compute_temporal_weights(edges_df, reference_date)

        # Apply burst-velocity amplification on top of temporal weights
        burst_weights = self._compute_burst_velocity_weights(edges_df, reference_date)
        temporal_weights = temporal_weights * burst_weights

        # Build directed graph with temporal weights
        G = nx.DiGraph()
        for idx, row in edges_df.iterrows():
            sender = str(row["Sender_account"])
            receiver = str(row["Receiver_account"])
            # Use .loc[idx] to access by label (works with any index, not just default 0-based)
            w = float(temporal_weights.loc[idx])
            # If edge already exists, accumulate temporal weight
            if G.has_edge(sender, receiver):
                G[sender][receiver]["temporal_weight"] += w
                G[sender][receiver]["original_weight"] += row["amount_local_npr"]
            else:
                G.add_edge(
                    sender, receiver,
                    temporal_weight=w,
                    original_weight=float(row["amount_local_npr"]),
                )

        nodes = list(G.nodes())
        N = len(nodes)

        if N == 0:
            return TDPageRankResult(
                scores={},
                normalized_scores={},
                cycle_member={},
                decay_impact={},
                converged=True,
                iterations=0,
                reference_date=reference_date,
            )

        # Create node-to-index mapping for deterministic ordering
        node_to_idx = {node: i for i, node in enumerate(sorted(nodes))}
        nodes_sorted = sorted(nodes)

        # Build transition matrix
        transition_matrix, dangling_mask = self._build_transition_matrix(
            G, nodes_sorted, node_to_idx
        )

        # Power iteration
        r = np.full(N, 1.0 / N, dtype=np.float64)
        converged = False
        iterations = 0

        d = self.damping
        teleport = (1.0 - d) / N

        for t in range(1, self.max_iter + 1):
            iterations = t

            # Dangling node contribution
            dangling_sum = np.sum(r[dangling_mask])

            # Core iteration: r_new = (1-d)/N + d * (M^T @ r) + d * dangling_sum / N
            r_new = teleport + d * (transition_matrix.T @ r) + d * dangling_sum / N

            # Convergence check: L1 norm
            l1_diff = np.sum(np.abs(r_new - r))
            if l1_diff < self.tol:
                converged = True
                r = r_new
                break

            r = r_new

        # Detect cycle nodes
        cycle_nodes = self._detect_cycle_nodes(G)

        # Apply cycle penalty
        r = self._apply_cycle_penalty(r, G, nodes_sorted, node_to_idx, cycle_nodes)

        # Detect dormant nodes (all incident edges > 30 days) and cap their scores
        # per Requirement 2.7: dormant node score <= 0.1 × max_score
        dormant_nodes = self._detect_dormant_nodes(G, edges_df, reference_date)
        if dormant_nodes:
            r_max_pre = np.max(r)
            dormancy_cap = 0.1 * r_max_pre
            for node in dormant_nodes:
                idx_node = node_to_idx[node]
                if r[idx_node] > dormancy_cap:
                    r[idx_node] = dormancy_cap

        # Build result dictionaries
        scores = {nodes_sorted[i]: float(r[i]) for i in range(N)}

        # Normalized scores: min-max to [0, 1]
        r_min = np.min(r)
        r_max = np.max(r)
        if r_max - r_min > 0:
            r_normalized = (r - r_min) / (r_max - r_min)
        else:
            # All scores equal
            r_normalized = np.zeros(N, dtype=np.float64)

        normalized_scores = {nodes_sorted[i]: float(r_normalized[i]) for i in range(N)}

        # Cycle membership flags
        cycle_member = {node: (node in cycle_nodes) for node in nodes_sorted}

        # Decay impact: 1 - (sum_decayed_weight / sum_original_weight) per node
        decay_impact = self._compute_decay_impact(G, nodes_sorted)

        return TDPageRankResult(
            scores=scores,
            normalized_scores=normalized_scores,
            cycle_member=cycle_member,
            decay_impact=decay_impact,
            converged=converged,
            iterations=iterations,
            reference_date=reference_date,
        )

    def _compute_temporal_weights(
        self, edges_df: pd.DataFrame, reference_date: date
    ) -> pd.Series:
        """
        Compute temporal edge weights: w_temporal(e) = amount × exp(-λ × Edge_Age).

        Edge_Age = (reference_date - edge.Date).days (non-negative integer)
        λ = 0.693 / half_life_days

        When log_amount_scale is True, amounts are replaced with log1p(amounts)
        before applying the decay formula. This prevents large single transactions
        from dominating the centrality computation — a known flaw in amount-weighted
        PageRank that prior art does NOT address.
        """
        dates = pd.to_datetime(edges_df["Date"])
        ref_dt = pd.Timestamp(reference_date)

        # Edge_Age in days (non-negative)
        edge_age = (ref_dt - dates).dt.days.astype(np.float64)
        edge_age = edge_age.clip(lower=0)

        amounts = edges_df["amount_local_npr"].astype(np.float64)

        # Novel mechanism: log-amount normalisation prevents single large transactions
        # from overwhelming the centrality signal (§ 103 differentiator)
        if self.log_amount_scale:
            amounts = np.log1p(amounts)

        # w_temporal = amount × exp(-λ × Edge_Age)
        temporal_weights = amounts * np.exp(-self.decay_lambda * edge_age)

        return temporal_weights

    def _compute_burst_velocity_weights(
        self, edges_df: pd.DataFrame, reference_date: date
    ) -> pd.Series:
        """
        Compute per-edge burst-velocity amplification multipliers.

        Novel mechanism: captures smurfing/structuring behaviour where a node
        suddenly sends many transactions in a short window.  Such bursts are a
        known AML red flag; amplifying their centrality signal (rather than
        penalising) makes the anomaly MORE visible to downstream scoring —
        a directional choice that distinguishes this from prior-art decay methods.

        Algorithm:
          1. For each sender node, count outgoing transactions in the last
             `burst_window_days` days (the "window count").
          2. If window_count > 5, that node is a "burst sender".
          3. For every outgoing edge from a burst sender, compute:
               burst_velocity_ratio = window_count / total_outgoing_count
               multiplier           = 1 + burst_velocity_ratio   (≥ 1.0)
          4. Non-burst edges receive multiplier = 1.0.

        Returns:
            pd.Series (same index as edges_df) of per-edge multipliers ≥ 1.0.
        """
        ref_dt = pd.Timestamp(reference_date)
        dates = pd.to_datetime(edges_df["Date"])
        senders = edges_df["Sender_account"].astype(str)

        edge_age = (ref_dt - dates).dt.days.astype(np.float64).clip(lower=0)
        in_window_mask = edge_age <= self.burst_window_days

        # Count outgoing edges per sender (total and within window)
        total_counts = senders.value_counts()  # total outgoing per sender
        window_counts = senders[in_window_mask].value_counts()  # within burst window

        # Identify burst senders (window_count > 5)
        burst_senders = set(window_counts[window_counts > 5].index)

        # Build per-edge multiplier Series (default 1.0)
        multipliers = pd.Series(1.0, index=edges_df.index, dtype=np.float64)

        if not burst_senders:
            return multipliers

        for idx, sender_val in zip(edges_df.index, edges_df["Sender_account"]):
            sender_str = str(sender_val)
            if sender_str in burst_senders:
                window_count = int(window_counts.get(sender_str, 0))
                total_count = int(total_counts.get(sender_str, 1))
                burst_velocity_ratio = window_count / total_count
                # Amplification (≥ 1.0): more burst → higher multiplier
                multipliers.at[idx] = 1.0 + burst_velocity_ratio

        return multipliers

    def _build_transition_matrix(
        self,
        G: nx.DiGraph,
        nodes_sorted: list,
        node_to_idx: Dict[str, int],
    ) -> tuple:
        """
        Build row-normalized transition matrix with dangling node handling.

        Returns:
            transition_matrix: N×N numpy array where M[i][j] = P(i → j)
                              (the probability of transitioning from node i to node j)
            dangling_mask: boolean array indicating dangling nodes (no outgoing edges)
        """
        N = len(nodes_sorted)
        M = np.zeros((N, N), dtype=np.float64)

        for i, node in enumerate(nodes_sorted):
            out_edges = G.out_edges(node, data=True)
            out_weights = []
            out_targets = []

            for _, target, data in out_edges:
                w = data.get("temporal_weight", 0.0)
                out_weights.append(w)
                out_targets.append(node_to_idx[target])

            total_out = sum(out_weights)
            if total_out > 0:
                for j, w in zip(out_targets, out_weights):
                    M[i][j] = w / total_out

        # Identify dangling nodes (rows that sum to 0)
        row_sums = M.sum(axis=1)
        dangling_mask = row_sums == 0.0

        return M, dangling_mask

    def _detect_cycle_nodes(self, G: nx.DiGraph) -> Set[str]:
        """
        Find nodes in strongly connected components of size > 2.
        """
        cycle_nodes = set()
        for scc in nx.strongly_connected_components(G):
            if len(scc) > 2:
                cycle_nodes.update(scc)
        return cycle_nodes

    def _apply_cycle_penalty(
        self,
        ranks: np.ndarray,
        G: nx.DiGraph,
        nodes_sorted: list,
        node_to_idx: Dict[str, int],
        cycle_nodes: Set[str],
    ) -> np.ndarray:
        """
        Apply cycle penalty when >80% of incident temporal weight is intra-SCC.

        Symmetric mode (asymmetric_scc_penalty=False, default):
            All qualifying SCC nodes: rank *= cycle_penalty (0.5×)

        Asymmetric mode (asymmetric_scc_penalty=True) — novel mechanism:
            Distinguishes role within the SCC based on directional flow:
            - Collector (in_scc_weight > out_scc_weight):
                  rank *= cycle_penalty × 0.5   (harder penalty, 0.25× at default)
            - Distributor (out_scc_weight > in_scc_weight):
                  rank *= cycle_penalty          (standard 0.5×)
            - Balanced (in_scc_weight ≈ out_scc_weight):
                  rank *= cycle_penalty × 0.75  (intermediate, 0.375× at default)

        This directional asymmetry is specific to financial money-laundering
        patterns: collectors accumulate funds; distributors disperse them.
        Prior art applies uniform cycle penalties without this distinction.
        """
        if not cycle_nodes:
            return ranks.copy()

        # Build SCC membership mapping for cycle nodes
        scc_map = {}  # node -> scc_id
        scc_id = 0
        for scc in nx.strongly_connected_components(G):
            if len(scc) > 2:
                for node in scc:
                    scc_map[node] = scc_id
                scc_id += 1

        ranks_penalized = ranks.copy()

        for node in cycle_nodes:
            idx = node_to_idx[node]
            node_scc_id = scc_map[node]

            total_weight = 0.0
            intra_scc_weight = 0.0
            out_scc_weight = 0.0   # outgoing intra-SCC temporal weight
            in_scc_weight = 0.0    # incoming intra-SCC temporal weight

            # Outgoing edges
            for _, target, data in G.out_edges(node, data=True):
                w = data.get("temporal_weight", 0.0)
                total_weight += w
                if target in scc_map and scc_map[target] == node_scc_id:
                    intra_scc_weight += w
                    out_scc_weight += w

            # Incoming edges
            for source, _, data in G.in_edges(node, data=True):
                w = data.get("temporal_weight", 0.0)
                total_weight += w
                if source in scc_map and scc_map[source] == node_scc_id:
                    intra_scc_weight += w
                    in_scc_weight += w

            # Apply penalty only when intra-SCC concentration exceeds 80%
            if total_weight > 0 and (intra_scc_weight / total_weight) > 0.80:
                if self.asymmetric_scc_penalty:
                    # Novel directional penalty (§ 103 differentiator)
                    if in_scc_weight > out_scc_weight:
                        # Collector: hardest penalty
                        effective_penalty = self.cycle_penalty * 0.5
                    elif out_scc_weight > in_scc_weight:
                        # Distributor: standard penalty
                        effective_penalty = self.cycle_penalty
                    else:
                        # Balanced: intermediate penalty
                        effective_penalty = self.cycle_penalty * 0.75
                else:
                    # Symmetric (backward-compatible) penalty
                    effective_penalty = self.cycle_penalty

                ranks_penalized[idx] *= effective_penalty

        return ranks_penalized

    def _detect_dormant_nodes(
        self,
        G: nx.DiGraph,
        edges_df: pd.DataFrame,
        reference_date: date,
        dormancy_threshold_days: int = 30,
    ) -> Set[str]:
        """
        Identify nodes where ALL incident edges (incoming and outgoing) have
        Edge_Age > dormancy_threshold_days.

        Per Requirement 2.7: such dormant nodes must have their rank score
        capped at 0.1 × max_rank to reflect dormancy.
        """
        dates = pd.to_datetime(edges_df["Date"])
        ref_ts = pd.Timestamp(reference_date)
        edge_ages = (ref_ts - dates).dt.days.values  # integer days per edge

        senders = edges_df["Sender_account"].astype(str).values
        receivers = edges_df["Receiver_account"].astype(str).values

        # Build a mapping: node -> list of edge ages for all incident edges
        node_edge_ages: Dict[str, list] = {node: [] for node in G.nodes()}
        for i in range(len(edges_df)):
            age = int(edge_ages[i])
            s = senders[i]
            r = receivers[i]
            if s in node_edge_ages:
                node_edge_ages[s].append(age)
            if r in node_edge_ages:
                node_edge_ages[r].append(age)

        dormant = set()
        for node, ages in node_edge_ages.items():
            if ages and all(a > dormancy_threshold_days for a in ages):
                dormant.add(node)
        return dormant

    def _compute_decay_impact(
        self, G: nx.DiGraph, nodes_sorted: list
    ) -> Dict[str, float]:
        """
        Compute decay_impact per node:
        decay_impact = 1 - (sum_decayed_weight / sum_original_weight)

        Uses all incident edges (both incoming and outgoing) for each node.
        """
        decay_impact = {}

        for node in nodes_sorted:
            total_original = 0.0
            total_decayed = 0.0

            # Outgoing edges
            for _, _, data in G.out_edges(node, data=True):
                total_original += data.get("original_weight", 0.0)
                total_decayed += data.get("temporal_weight", 0.0)

            # Incoming edges
            for _, _, data in G.in_edges(node, data=True):
                total_original += data.get("original_weight", 0.0)
                total_decayed += data.get("temporal_weight", 0.0)

            if total_original > 0:
                decay_impact[node] = 1.0 - (total_decayed / total_original)
            else:
                decay_impact[node] = 0.0

        return decay_impact


if __name__ == "__main__":
    # Quick demo / manual test
    from datetime import timedelta

    # Create sample edges
    ref_date = date(2024, 1, 1)
    data = [
        {"Sender_account": "A", "Receiver_account": "B", "amount_local_npr": 50000.0, "Date": ref_date - timedelta(days=2)},
        {"Sender_account": "B", "Receiver_account": "C", "amount_local_npr": 30000.0, "Date": ref_date - timedelta(days=5)},
        {"Sender_account": "C", "Receiver_account": "A", "amount_local_npr": 25000.0, "Date": ref_date - timedelta(days=3)},
        {"Sender_account": "A", "Receiver_account": "D", "amount_local_npr": 100000.0, "Date": ref_date - timedelta(days=1)},
        {"Sender_account": "D", "Receiver_account": "E", "amount_local_npr": 75000.0, "Date": ref_date - timedelta(days=10)},
    ]
    edges_df = pd.DataFrame(data)
    edges_df["Date"] = pd.to_datetime(edges_df["Date"])

    engine = TDPageRankEngine()
    result = engine.compute(edges_df, reference_date=ref_date)

    print("TD-PageRank Results:")
    print(f"  Converged: {result.converged} in {result.iterations} iterations")
    print(f"  Reference Date: {result.reference_date}")
    print("\n  Raw Scores:")
    for node, score in sorted(result.scores.items()):
        print(f"    {node}: {score:.8f}")
    print("\n  Normalized Scores:")
    for node, score in sorted(result.normalized_scores.items()):
        print(f"    {node}: {score:.6f}")
    print("\n  Cycle Members:")
    for node, flag in sorted(result.cycle_member.items()):
        print(f"    {node}: {flag}")
    print("\n  Decay Impact:")
    for node, impact in sorted(result.decay_impact.items()):
        print(f"    {node}: {impact:.6f}")
