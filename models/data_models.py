"""
Core data model classes for the Patentable AML Innovations.

Contains dataclasses for:
- TD-PageRank Engine outputs
- Topology-Adaptive Fusion models
- Degradation Controller models
- Patent Evaluation models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import date, datetime
from enum import Enum
import numpy as np


# === TD-PageRank Models ===

@dataclass
class TDPageRankResult:
    """Output of the TD-PageRank computation."""
    scores: Dict[str, float]              # Raw TD-PageRank scores per node
    normalized_scores: Dict[str, float]   # Min-max normalized to [0, 1]
    cycle_member: Dict[str, bool]         # SCC membership flag per node
    decay_impact: Dict[str, float]        # 1 - (decayed_weight / original_weight) per node
    converged: bool                        # Whether power iteration converged
    iterations: int                        # Number of iterations performed
    reference_date: date                   # The date used for Edge_Age computation


# === Topology-Adaptive Fusion Models ===

@dataclass
class TopologyVector:
    """5-dimensional descriptor of an ego-network's structure."""
    edge_density: float           # |E| / (|V| * (|V|-1)) for directed graph
    diameter: int                 # Longest shortest path in ego-network (undirected)
    avg_clustering: float         # Average clustering coefficient of ego-network nodes
    degree_asymmetry: float       # var(in_degree) / var(out_degree), 0 if denom is 0
    component_ratio: float        # largest_component_size / total_nodes

    def to_tensor(self) -> 'torch.Tensor':
        """Convert to a 1D float tensor for MLP input."""
        import torch
        return torch.tensor(
            [self.edge_density, self.diameter, self.avg_clustering,
             self.degree_asymmetry, self.component_ratio],
            dtype=torch.float32
        )


@dataclass
class FusionResult:
    """Output of the adaptive fusion for a single account."""
    fused_score: float                    # Final score in [0.0, 1.0]
    weights: Tuple[float, float, float]   # (w_ml, w_graph, w_rules)
    fallback_triggered: bool              # Whether static fallback was used
    fallback_reason: Optional[str]        # Reason for fallback (if triggered)
    topology_vector: Optional[TopologyVector]  # The computed topology (None on fallback)


# === Degradation Controller Models ===

class ComponentStatus(Enum):
    """Status of a pipeline component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    COOLDOWN = "cooldown"  # Locked degraded due to flapping


@dataclass
class HealthVector:
    """Real-time health metrics for a single pipeline component."""
    heartbeat_latency_ms: float    # ms since last successful response
    kl_divergence: float           # Output distribution divergence from reference
    throughput_ratio: float        # current_tps / baseline_tps (rolling 60-min avg)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionPath:
    """A pre-computed scoring route through the pipeline."""
    path_id: str                              # e.g., "full", "no_pagerank", "lgbm_only"
    required_components: Set[str]             # Components needed for this path
    measured_precision_at_50: float           # Precision@50 from offline evaluation
    description: str                          # Human-readable description


class DegradationLevel(Enum):
    """Enumeration of system degradation levels."""
    LEVEL_0 = "full_system"           # All components healthy
    LEVEL_1 = "single_component"     # One component degraded
    LEVEL_2 = "dual_component"       # Two components degraded
    LEVEL_3 = "lgbm_pagerank_only"   # Only ML + graph
    LEVEL_4 = "lgbm_rules_only"      # Only ML + rules
    LEVEL_5 = "lgbm_only"           # Minimal: ML model only


@dataclass
class RoutingDecision:
    """Result of the degradation controller's routing logic."""
    action: str                           # "switch_path" | "halt" | "maintain"
    selected_path: Optional[ExecutionPath]
    degraded_components: Set[str]
    healthy_components: Set[str]
    precision_met: bool                   # Whether precision budget is satisfied
    stale_queue_available: bool           # Whether last valid queue can be served
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StateTransitionLog:
    """Immutable log entry for component state changes."""
    timestamp: datetime
    component: str
    from_status: ComponentStatus
    to_status: ComponentStatus
    trigger_metric: str                   # e.g., "heartbeat_latency_ms=6200"
    selected_path: str                    # path_id after transition


# === Patent Evaluation Models ===

@dataclass
class FusionNoveltyReport:
    """Quantifies Topology-Adaptive Fusion improvement over static fusion."""
    adaptive_precision_at_50: float
    static_precision_at_50: float
    absolute_improvement: float           # adaptive - static
    relative_improvement_pct: float       # (adaptive - static) / static * 100
    meets_novelty_threshold: bool         # relative_improvement >= 10%
    prior_art_ref: str = "US20220405860"


@dataclass
class PageRankNoveltyReport:
    """Quantifies TD-PageRank differentiation from standard PageRank."""
    mean_absolute_difference: float       # avg |td_score - std_score| across all nodes
    mean_absolute_pct_difference: float   # avg |td - std| / std across all nodes
    meets_novelty_threshold: bool         # mean_absolute_difference >= 0.01
    prior_art_ref: str = "US20240062041"


@dataclass
class DegradationNoveltyReport:
    """Quantifies Degradation Controller precision maintenance."""
    path_precisions: Dict[str, float]     # path_id -> measured Precision@50
    min_precision_maintained: float       # Lowest precision across all paths
    meets_precision_budget: bool          # All paths >= 0.60
    prior_art_ref: str = "US20260038036"


@dataclass
class PatentMetricsReport:
    """Complete patent evaluation report."""
    fusion_report: FusionNoveltyReport
    pagerank_report: PageRankNoveltyReport
    degradation_report: DegradationNoveltyReport
    dataset_size: int
    temporal_split_date: date
    random_seed: int
    generated_at: datetime = field(default_factory=datetime.now)
