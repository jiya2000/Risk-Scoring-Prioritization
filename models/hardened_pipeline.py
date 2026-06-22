"""
Hardened Pipeline Factory — Wires all architecture-hardened components into
a cohesive pipeline configuration based on ArchitectureHardeningConfig flags.

Data Flow (when fully enabled):
    graph edges → TDPageRankEngine (with LearnableSCCPenalty)
        → AdaptiveFusionEngine (with TopologyEmbeddingNetwork + IsolationDetector + DeepTopologyAttentionGate)
            → DegradationController (with OnlinePrecisionMonitor + EnhancedConceptDriftDetector + EnhancedAdaptivePrecisionBudget)
                → ResourceManager (logging all path switches, memory operations, and resource utilization)

When ArchitectureHardeningConfig() uses defaults (all False), the pipeline behaves
identically to the original — all hardened components are None and the system uses
static heuristic penalties, manual TopologyVector, and offline precision assumptions.

Validates: Requirements 1.5, 2.7, 7.5
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from dataclasses import dataclass, field
from typing import Optional

from models.architecture_config import ArchitectureHardeningConfig

logger = logging.getLogger(__name__)


@dataclass
class HardenedPipelineComponents:
    """
    Container holding references to all hardened pipeline components.

    Components are None when their corresponding config flag is False,
    enabling graceful degradation and backward-compatible operation.
    """
    config: ArchitectureHardeningConfig

    # TD-PageRank hardening (Requirement 1)
    learnable_scc_penalty: Optional[object] = None

    # Topology Embedding (Requirement 2)
    topology_embedding_net: Optional[object] = None

    # Isolation Detector (Requirement 3)
    isolation_detector: Optional[object] = None

    # Deep Fusion Gate (Requirement 6)
    deep_gate: Optional[object] = None

    # Online Precision Monitor (Requirement 4)
    online_precision_monitor: Optional[object] = None

    # Enhanced Concept Drift Detector (Requirement 5)
    enhanced_drift_detector: Optional[object] = None

    # Enhanced Adaptive Precision Budget (Requirement 8)
    enhanced_budget: Optional[object] = None

    # Resource Manager (Requirement 7)
    resource_manager: Optional[object] = None

    # Summary of what was activated
    activated_components: list = field(default_factory=list)

    @property
    def any_active(self) -> bool:
        """True if at least one hardened component is active."""
        return len(self.activated_components) > 0


def create_hardened_pipeline(
    config: Optional[ArchitectureHardeningConfig] = None,
) -> HardenedPipelineComponents:
    """
    Factory function that instantiates all hardened components based on config flags.

    Components with use_*=False are left as None (graceful degradation).
    All imports are guarded with try/except so the pipeline still works even if
    optional dependencies (PyTorch, PyTorch Geometric) are unavailable.

    Args:
        config: ArchitectureHardeningConfig instance. If None, defaults are used
                (all flags False = fully backward compatible).

    Returns:
        HardenedPipelineComponents with references to all instantiated components.
    """
    if config is None:
        config = ArchitectureHardeningConfig()

    components = HardenedPipelineComponents(config=config)

    # ─────────────────────────────────────────────────────────────────────
    # Resource Manager (Requirement 7) — instantiated first since other
    # components may reference it for resource tracking
    # ─────────────────────────────────────────────────────────────────────
    if config.enable_resource_tracking:
        try:
            from models.resource_manager import ResourceManager
            components.resource_manager = ResourceManager()
            components.activated_components.append("resource_manager")
            logger.info("HardenedPipeline: ResourceManager activated.")
        except Exception as e:
            logger.warning(f"HardenedPipeline: ResourceManager unavailable: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Learnable SCC Penalty (Requirement 1)
    # ─────────────────────────────────────────────────────────────────────
    if config.use_learnable_scc_penalty:
        try:
            import torch
            from models.td_pagerank import LearnableSCCPenalty

            penalty_model = LearnableSCCPenalty(
                input_dim=5,
                hidden_dim=config.scc_penalty_hidden_dim,
            )

            # Load pre-trained weights if path is specified
            if config.scc_penalty_model_path and os.path.exists(config.scc_penalty_model_path):
                state_dict = torch.load(config.scc_penalty_model_path, map_location="cpu")
                penalty_model.load_state_dict(state_dict)
                logger.info(
                    f"HardenedPipeline: LearnableSCCPenalty loaded weights from "
                    f"{config.scc_penalty_model_path}"
                )

            penalty_model.eval()
            components.learnable_scc_penalty = penalty_model
            components.activated_components.append("learnable_scc_penalty")
            logger.info("HardenedPipeline: LearnableSCCPenalty activated.")
        except Exception as e:
            logger.warning(f"HardenedPipeline: LearnableSCCPenalty unavailable: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Topology Embedding Network (Requirement 2)
    # ─────────────────────────────────────────────────────────────────────
    if config.use_gnn_topology:
        try:
            from models.adaptive_fusion import TopologyEmbeddingNetwork

            topology_net = TopologyEmbeddingNetwork(
                node_feature_dim=4,
                hidden_dim=config.gnn_hidden_dim,
                embedding_dim=config.gnn_embedding_dim,
                num_layers=config.gnn_num_layers,
                timeout_ms=config.gnn_timeout_ms,
            )
            topology_net.eval()
            components.topology_embedding_net = topology_net
            components.activated_components.append("topology_embedding_net")
            logger.info("HardenedPipeline: TopologyEmbeddingNetwork activated.")
        except Exception as e:
            logger.warning(f"HardenedPipeline: TopologyEmbeddingNetwork unavailable: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Isolation Detector (Requirement 3)
    # ─────────────────────────────────────────────────────────────────────
    if config.use_learned_isolation:
        try:
            from models.adaptive_fusion import IsolationDetector

            isolation = IsolationDetector(
                embedding_dim=config.gnn_embedding_dim,
                hidden_dim=config.isolation_hidden_dim,
            )
            isolation.eval()
            components.isolation_detector = isolation
            components.activated_components.append("isolation_detector")
            logger.info("HardenedPipeline: IsolationDetector activated.")
        except Exception as e:
            logger.warning(f"HardenedPipeline: IsolationDetector unavailable: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Deep Fusion Gate (Requirement 6)
    # ─────────────────────────────────────────────────────────────────────
    if config.use_deep_gate:
        try:
            from models.adaptive_fusion import DeepTopologyAttentionGate

            # Input dim matches topology embedding output or manual vector size
            input_dim = config.gnn_embedding_dim if config.use_gnn_topology else 5
            deep_gate = DeepTopologyAttentionGate(
                input_dim=input_dim,
                hidden_dim=config.deep_gate_hidden_dim,
                num_hidden_layers=config.deep_gate_num_layers,
                dropout_rate=config.deep_gate_dropout,
            )
            deep_gate.eval()
            components.deep_gate = deep_gate
            components.activated_components.append("deep_gate")
            logger.info("HardenedPipeline: DeepTopologyAttentionGate activated.")
        except Exception as e:
            logger.warning(f"HardenedPipeline: DeepTopologyAttentionGate unavailable: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Online Precision Monitor (Requirement 4)
    # ─────────────────────────────────────────────────────────────────────
    if config.use_online_precision:
        try:
            from models.degradation_controller import OnlinePrecisionMonitor

            components.online_precision_monitor = OnlinePrecisionMonitor(
                min_samples=config.online_precision_min_samples,
                window_size=config.online_precision_window,
            )
            components.activated_components.append("online_precision_monitor")
            logger.info("HardenedPipeline: OnlinePrecisionMonitor activated.")
        except Exception as e:
            logger.warning(f"HardenedPipeline: OnlinePrecisionMonitor unavailable: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Enhanced Concept Drift Detector (Requirement 5)
    # ─────────────────────────────────────────────────────────────────────
    if config.use_enhanced_drift:
        try:
            from models.degradation_controller import (
                EnhancedConceptDriftDetector,
                EnhancedAdaptivePrecisionBudget,
            )

            # Enhanced budget with online + shadow weighting
            enhanced_budget = EnhancedAdaptivePrecisionBudget(
                base_budget=0.60,
                min_budget=0.55,
                max_budget=0.75,
                online_weight=config.online_budget_weight,
                shadow_weight=config.shadow_budget_weight,
            )
            components.enhanced_budget = enhanced_budget
            components.activated_components.append("enhanced_budget")

            # Enhanced drift detector wired to precision monitor and budget
            components.enhanced_drift_detector = EnhancedConceptDriftDetector(
                kl_threshold=0.15,
                precision_monitor=components.online_precision_monitor,
                precision_budget=enhanced_budget,
            )
            components.activated_components.append("enhanced_drift_detector")
            logger.info("HardenedPipeline: EnhancedConceptDriftDetector + EnhancedAdaptivePrecisionBudget activated.")
        except Exception as e:
            logger.warning(f"HardenedPipeline: Enhanced drift detection unavailable: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Log summary
    # ─────────────────────────────────────────────────────────────────────
    if components.any_active:
        logger.info(
            f"HardenedPipeline: {len(components.activated_components)} components activated: "
            f"{components.activated_components}"
        )
    else:
        logger.info("HardenedPipeline: All hardening flags are False — using baseline pipeline.")

    return components


def wire_td_pagerank_engine(components: HardenedPipelineComponents, **engine_kwargs):
    """
    Create a TDPageRankEngine instance wired with hardened components.

    If learnable SCC penalty is active, passes it to the engine along with
    the resource manager for penalty buffer allocation logging.

    Args:
        components: HardenedPipelineComponents from create_hardened_pipeline().
        **engine_kwargs: Additional keyword arguments for TDPageRankEngine().

    Returns:
        TDPageRankEngine instance, or None if the module cannot be imported.
    """
    try:
        from models.td_pagerank import TDPageRankEngine

        kwargs = dict(engine_kwargs)

        # Wire learnable penalty if activated
        if components.learnable_scc_penalty is not None:
            kwargs["use_learnable_penalty"] = True
            kwargs["learnable_penalty_model"] = components.learnable_scc_penalty
        else:
            kwargs.setdefault("use_learnable_penalty", False)

        # Wire resource manager if activated
        if components.resource_manager is not None:
            kwargs["resource_manager"] = components.resource_manager

        engine = TDPageRankEngine(**kwargs)
        logger.info(
            f"HardenedPipeline: TDPageRankEngine created "
            f"(learnable_penalty={components.learnable_scc_penalty is not None}, "
            f"resource_manager={components.resource_manager is not None})"
        )
        return engine

    except Exception as e:
        logger.warning(f"HardenedPipeline: Could not create TDPageRankEngine: {e}")
        return None


def wire_adaptive_fusion_engine(components: HardenedPipelineComponents, **engine_kwargs):
    """
    Create an AdaptiveFusionEngine instance wired with hardened components.

    Passes topology embedding net, isolation detector, deep gate, resource manager,
    and the config to the engine based on which components are active.

    Args:
        components: HardenedPipelineComponents from create_hardened_pipeline().
        **engine_kwargs: Additional keyword arguments for AdaptiveFusionEngine().

    Returns:
        AdaptiveFusionEngine instance, or None if the module cannot be imported.
    """
    try:
        from models.adaptive_fusion import AdaptiveFusionEngine

        kwargs = dict(engine_kwargs)

        # Wire hardening config
        kwargs["hardening_config"] = components.config

        # Wire topology embedding if activated
        if components.topology_embedding_net is not None:
            kwargs["topology_embedding_net"] = components.topology_embedding_net

        # Wire isolation detector if activated
        if components.isolation_detector is not None:
            kwargs["isolation_detector"] = components.isolation_detector

        # Wire deep gate if activated
        if components.deep_gate is not None:
            kwargs["deep_gate"] = components.deep_gate

        # Wire resource manager if activated
        if components.resource_manager is not None:
            kwargs["resource_manager"] = components.resource_manager

        engine = AdaptiveFusionEngine(**kwargs)
        logger.info(
            f"HardenedPipeline: AdaptiveFusionEngine created "
            f"(gnn={components.topology_embedding_net is not None}, "
            f"isolation={components.isolation_detector is not None}, "
            f"deep_gate={components.deep_gate is not None})"
        )
        return engine

    except Exception as e:
        logger.warning(f"HardenedPipeline: Could not create AdaptiveFusionEngine: {e}")
        return None


def wire_degradation_controller(components: HardenedPipelineComponents, **controller_kwargs):
    """
    Create a DegradationController instance wired with hardened components.

    Passes the online precision monitor, enhanced drift detector, and resource
    manager to the controller based on which components are active.

    The EnhancedAdaptivePrecisionBudget is a standalone component that works
    alongside the controller (not passed as a constructor arg). It is available
    via components.enhanced_budget for external callers to use.

    Args:
        components: HardenedPipelineComponents from create_hardened_pipeline().
        **controller_kwargs: Additional keyword arguments for DegradationController().

    Returns:
        DegradationController instance, or None if the module cannot be imported.
    """
    try:
        from models.degradation_controller import DegradationController

        kwargs = dict(controller_kwargs)

        # Wire online precision monitor if activated
        if components.online_precision_monitor is not None:
            kwargs["online_precision_monitor"] = components.online_precision_monitor

        # Wire enhanced drift detector if activated
        if components.enhanced_drift_detector is not None:
            kwargs["enhanced_drift_detector"] = components.enhanced_drift_detector

        # Wire resource manager if activated
        if components.resource_manager is not None:
            kwargs["resource_manager"] = components.resource_manager

        controller = DegradationController(**kwargs)
        logger.info(
            f"HardenedPipeline: DegradationController created "
            f"(online_precision={components.online_precision_monitor is not None}, "
            f"enhanced_drift={components.enhanced_drift_detector is not None}, "
            f"enhanced_budget={components.enhanced_budget is not None}, "
            f"resource_manager={components.resource_manager is not None})"
        )
        return controller

    except Exception as e:
        logger.warning(f"HardenedPipeline: Could not create DegradationController: {e}")
        return None


def print_pipeline_status(components: HardenedPipelineComponents) -> None:
    """Print a human-readable summary of which hardened components are active."""
    print("\n  ┌─── Architecture Hardening Status ───────────────────────────────┐")
    flags = [
        ("Learnable SCC Penalty", components.config.use_learnable_scc_penalty, components.learnable_scc_penalty),
        ("GNN Topology Embedding", components.config.use_gnn_topology, components.topology_embedding_net),
        ("Learned Isolation Detector", components.config.use_learned_isolation, components.isolation_detector),
        ("Deep Fusion Gate", components.config.use_deep_gate, components.deep_gate),
        ("Online Precision Monitor", components.config.use_online_precision, components.online_precision_monitor),
        ("Enhanced Drift Detection", components.config.use_enhanced_drift, components.enhanced_drift_detector),
        ("Resource Tracking (§101)", components.config.enable_resource_tracking, components.resource_manager),
    ]
    for name, flag, component in flags:
        if flag and component is not None:
            status = "✅ ACTIVE"
        elif flag and component is None:
            status = "⚠️ ENABLED but unavailable (dependency missing)"
        else:
            status = "⬚  disabled (default)"
        print(f"  │  {name:<32} {status}")
    print("  └────────────────────────────────────────────────────────────────┘")
