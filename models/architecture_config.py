"""
Central configuration dataclass for all Architecture Hardening features.

Controls feature flags and hyperparameters for:
- Learnable SCC Penalty (TD-PageRank)
- GNN Topology Embedding (Adaptive Fusion)
- Learned Isolation Detector
- Deep Fusion Gate
- Online Precision Monitor (Degradation Controller)
- Enhanced Drift Detection
- Adaptive Budget weighting
- Resource Management (Patent § 101)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Optional


@dataclass
class ArchitectureHardeningConfig:
    """Central configuration for all hardening features.

    All new components default to disabled (False) to maintain backward
    compatibility with existing behavior per Requirements 1.5 and 2.7.
    """

    # SCC Penalty
    use_learnable_scc_penalty: bool = False
    scc_penalty_hidden_dim: int = 32
    scc_penalty_model_path: Optional[str] = None

    # Topology Embedding
    use_gnn_topology: bool = False
    gnn_embedding_dim: int = 16
    gnn_hidden_dim: int = 32
    gnn_num_layers: int = 2
    gnn_timeout_ms: float = 200.0

    # Isolation Detector
    use_learned_isolation: bool = False
    isolation_hidden_dim: int = 32

    # Deep Fusion Gate
    use_deep_gate: bool = False
    deep_gate_hidden_dim: int = 64
    deep_gate_num_layers: int = 2
    deep_gate_dropout: float = 0.1

    # Online Precision Monitor
    use_online_precision: bool = False
    online_precision_min_samples: int = 200
    online_precision_window: int = 500

    # Enhanced Drift Detection
    use_enhanced_drift: bool = False

    # Adaptive Budget
    online_budget_weight: float = 0.7
    shadow_budget_weight: float = 0.3

    # Resource Management
    enable_resource_tracking: bool = False
