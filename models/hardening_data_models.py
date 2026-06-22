"""
Data models for Architecture Hardening components.

Contains dataclasses for inter-component communication:
- SCCFlowFeatures: Per-node feature vector for learnable SCC penalty
- TopologyEmbedding: Result of GNN-based ego-network embedding
- IsolationResult: Output of the learned isolation detector
- PrecisionEstimate: A single precision estimate from the Online Precision Monitor
- DriftClassification: Result of enhanced drift detection
- ResourceUtilizationSnapshot: Snapshot of resource usage for patent § 101 evidence
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime

import numpy as np


@dataclass
class SCCFlowFeatures:
    """Per-node feature vector for learnable SCC penalty.

    Captures intra-SCC flow characteristics used as input to the
    LearnableSCCPenalty MLP.
    """
    intra_inflow_weight: float       # Sum of temporal weights on intra-SCC incoming edges
    intra_outflow_weight: float      # Sum of temporal weights on intra-SCC outgoing edges
    weight_ratio: float              # intra_inflow / (intra_inflow + intra_outflow)
    scc_size: int                    # Number of nodes in the SCC
    node_degree_in_scc: int          # Degree of this node within the SCC subgraph


@dataclass
class TopologyEmbedding:
    """Result of GNN-based ego-network embedding.

    Wraps the embedding vector along with metadata about the inference
    (timing, fallback status, ego-network size).
    """
    embedding: np.ndarray            # Shape (embedding_dim,) — default 16-d
    inference_time_ms: float         # Wall-clock inference time
    fallback_used: bool              # Whether manual TopologyVector was used instead
    ego_network_size: int            # Number of nodes in the 2-hop ego-network


@dataclass
class IsolationResult:
    """Output of the learned isolation detector.

    Contains the continuous isolation score and the adjusted fusion weights
    that respect the constraint invariants (each weight in [0.05, 0.90],
    sum to 1.0).
    """
    isolation_score: float           # Continuous score in [0.0, 1.0]
    adjusted_weights: Tuple[float, float, float]  # (w_ml, w_graph, w_rules)
    weight_constraint_satisfied: bool  # All in [0.05, 0.90] and sum == 1.0


@dataclass
class PrecisionEstimate:
    """A single precision estimate from the Online Precision Monitor.

    Tracks per-path precision with sample count and confidence interval
    for audit purposes.
    """
    path_id: str
    precision_at_50: float
    sample_count: int                # Number of labeled samples used
    timestamp: datetime
    confidence_interval: Tuple[float, float]  # 95% CI


@dataclass
class DriftClassification:
    """Result of enhanced drift detection.

    Classifies whether a detected distribution shift corresponds to
    actual precision degradation or is a benign shift.
    """
    classification: str              # 'no_drift' | 'benign_shift' | 'precision_degraded'
    kl_divergence: float
    estimated_precision: Optional[float]
    path_id: str
    timestamp: datetime


@dataclass
class ResourceUtilizationSnapshot:
    """Snapshot of resource usage for patent § 101 evidence.

    Logs concrete resource management operations (memory allocation,
    computation time, processing unit utilization) tied to execution
    path switches.
    """
    timestamp: datetime
    active_path_id: str
    memory_allocated_bytes: int
    memory_released_bytes: int
    computation_time_ms: float
    active_component_count: int
    event_type: str                  # 'path_switch' | 'model_swap' | 'degradation'
