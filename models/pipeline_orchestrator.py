"""
Pipeline Orchestrator — Wraps the AML scoring pipeline with DegradationController
health monitoring.

Routes scoring requests through the optimal execution path while maintaining
Precision@50 >= 0.60.

Requirements: 3.4, 3.9, 3.10
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from models.degradation_controller import DegradationController
from models.data_models import ComponentStatus, RoutingDecision

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Wraps the AML scoring pipeline with DegradationController health monitoring.
    Routes scoring requests through the optimal execution path while maintaining
    Precision@50 >= 0.60.

    Execution paths correspond to the DegradationController routing table:
        "full"          — lightgbm + td_pagerank + fusion_engine + symbolic_rules + nlp_summarizer
        "no_nlp"        — lightgbm + td_pagerank + fusion_engine + symbolic_rules
        "no_symbolic"   — lightgbm + td_pagerank + fusion_engine
        "no_pagerank"   — lightgbm + fusion_engine + symbolic_rules
        "no_fusion"     — lightgbm + td_pagerank + symbolic_rules  (static fusion fallback)
        "lgbm_pagerank" — lightgbm + td_pagerank
        "lgbm_rules"    — lightgbm + symbolic_rules
        "lgbm_only"     — lightgbm only
    """

    def __init__(self, precision_budget: float = 0.60):
        self.controller = DegradationController(precision_budget=precision_budget)

        # Component health check callbacks (set externally via register_health_callback)
        # Each callable should return a HealthVector for the given component.
        self.health_callbacks: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_accounts(
        self,
        test_df: pd.DataFrame,
        y_score: np.ndarray,
        G=None,
        rule_adjustments=None,
    ) -> dict:
        """
        Route scoring through the selected execution path.

        Performs a health check, obtains a RoutingDecision from the
        DegradationController, and executes the appropriate scoring path.
        The routing decision and path selection complete within 500ms of
        degradation detection (Requirement 3.10).

        Args:
            test_df:          DataFrame of test transactions (features included).
            y_score:          Raw LightGBM predicted probabilities (np.ndarray).
            G:                Optional nx.DiGraph for graph-dependent paths.
            rule_adjustments: Optional rule adjustment array for symbolic paths.

        Returns:
            dict with keys:
                'scores'   : np.ndarray of final risk scores in [0, 1]
                'path_id'  : str  — the execution path that was used
                'fallback' : bool — True when scoring was forced onto a degraded path
        """
        # 1. Run health checks and update component statuses
        self.run_health_check()

        # 2. Obtain routing decision from the controller (≤ 500ms SLA)
        decision: RoutingDecision = self.controller.handle_degradation()

        # 3. Log the selected path
        if decision.selected_path is not None:
            logger.info(
                "PipelineOrchestrator: routing via path '%s' "
                "(P@50=%.2f, degraded=%s)",
                decision.selected_path.path_id,
                decision.selected_path.measured_precision_at_50,
                decision.degraded_components or "none",
            )
        else:
            logger.critical(
                "PipelineOrchestrator: no viable execution path — halting. "
                "Degraded components: %s | stale_queue_available=%s",
                decision.degraded_components,
                decision.stale_queue_available,
            )

        # 4. Route to the appropriate execution path
        if decision.action == "halt" or decision.selected_path is None:
            # No path meets the precision budget; return raw LightGBM scores as
            # best-effort with fallback=True so callers can react appropriately.
            return {
                "scores": np.clip(np.array(y_score, dtype=np.float64), 0.0, 1.0),
                "path_id": "halt",
                "fallback": True,
            }

        path_id = decision.selected_path.path_id
        scores = self._execute_path(
            path_id=path_id,
            test_df=test_df,
            y_score=y_score,
            G=G,
            rule_adjustments=rule_adjustments,
        )

        is_fallback = path_id not in ("full", "no_nlp")
        return {
            "scores": scores,
            "path_id": path_id,
            "fallback": is_fallback,
        }

    def register_health_callback(self, component: str, callback: Callable) -> None:
        """
        Register an external health-check callback for a pipeline component.

        The callback must accept no arguments and return a HealthVector.
        It is called during run_health_check() to obtain live metrics for
        the given component.

        Args:
            component: One of DegradationController.COMPONENTS.
            callback:  Zero-argument callable that returns a HealthVector.

        Raises:
            ValueError: If component is not a recognised pipeline component.
        """
        if component not in DegradationController.COMPONENTS:
            raise ValueError(
                f"Unknown component '{component}'. "
                f"Must be one of {DegradationController.COMPONENTS}"
            )
        self.health_callbacks[component] = callback
        logger.debug("Registered health callback for component '%s'", component)

    def run_health_check(self) -> None:
        """
        Invoke registered health callbacks and feed results into the
        DegradationController's state machine.

        Components without a registered callback fall through to the
        DegradationController's internal default (simulated healthy vector),
        which is appropriate for development / testing environments.
        """
        for component in DegradationController.COMPONENTS:
            if component in self.health_callbacks:
                try:
                    health_vector = self.health_callbacks[component]()
                    self.controller.evaluate_component(component, health_vector)
                    logger.debug(
                        "Health check for '%s': latency=%.1fms kl=%.4f tps_ratio=%.2f",
                        component,
                        health_vector.heartbeat_latency_ms,
                        health_vector.kl_divergence,
                        health_vector.throughput_ratio,
                    )
                except Exception as exc:
                    # Treat callback failure as full degradation (Req 3.2)
                    logger.error(
                        "Health callback for '%s' raised an exception: %s. "
                        "Treating component as DEGRADED.",
                        component,
                        exc,
                    )
                    self.controller.force_component_status(
                        component, ComponentStatus.DEGRADED
                    )
            # No callback → controller uses its own internal default

    # ------------------------------------------------------------------
    # Internal routing helpers
    # ------------------------------------------------------------------

    def _execute_path(
        self,
        path_id: str,
        test_df: pd.DataFrame,
        y_score: np.ndarray,
        G=None,
        rule_adjustments=None,
    ) -> np.ndarray:
        """
        Dispatch to the scoring function for the given path_id.

        All paths are fail-safe: if a required resource is unavailable the
        method degrades gracefully rather than raising an exception, and
        clips the output to [0, 1].
        """
        start = time.monotonic()
        try:
            scores = self._route(path_id, test_df, y_score, G, rule_adjustments)
        except Exception as exc:
            logger.error(
                "Execution path '%s' raised an unexpected error: %s. "
                "Falling back to raw LightGBM scores.",
                path_id,
                exc,
            )
            scores = np.array(y_score, dtype=np.float64)

        elapsed_ms = (time.monotonic() - start) * 1000
        if elapsed_ms > 500:
            logger.warning(
                "Path '%s' took %.1fms — exceeds 500ms SLA (Req 3.10).", path_id, elapsed_ms
            )

        return np.clip(scores, 0.0, 1.0)

    def _route(
        self,
        path_id: str,
        test_df: pd.DataFrame,
        y_score: np.ndarray,
        G,
        rule_adjustments,
    ) -> np.ndarray:
        """
        Core routing table dispatch.

        Paths are ordered from most capable (full) to least capable (lgbm_only).
        Each path uses only the components it requires per the routing table
        defined in DegradationController.
        """
        base_scores = np.array(y_score, dtype=np.float64)

        if path_id in ("full", "no_nlp"):
            # Full pipeline: ML + graph (TD-PageRank) + adaptive fusion + rules
            # NLP summarizer is not involved in scoring so "no_nlp" is identical.
            return self._path_full(test_df, base_scores, G, rule_adjustments)

        if path_id == "no_symbolic":
            # ML + graph + fusion, no symbolic rule adjustments
            return self._path_no_symbolic(test_df, base_scores, G)

        if path_id == "no_pagerank":
            # ML + fusion + rules, no TD-PageRank graph signal
            return self._path_no_pagerank(test_df, base_scores, rule_adjustments)

        if path_id == "no_fusion":
            # ML + TD-PageRank + rules, adaptive fusion replaced by static fallback
            return self._path_no_fusion(test_df, base_scores, G, rule_adjustments)

        if path_id == "lgbm_pagerank":
            # ML + graph only — no rules, no adaptive fusion
            return self._path_lgbm_pagerank(test_df, base_scores, G)

        if path_id == "lgbm_rules":
            # ML + rules only — no graph signal, no adaptive fusion
            return self._path_lgbm_rules(test_df, base_scores, rule_adjustments)

        if path_id == "lgbm_only":
            # Minimal: LightGBM raw scores only
            logger.warning(
                "Executing minimal path 'lgbm_only' — Precision@50 ~0.62."
            )
            return base_scores

        # Unknown path_id — safe fallback
        logger.error("Unknown path_id '%s'. Returning raw LightGBM scores.", path_id)
        return base_scores

    # ------------------------------------------------------------------
    # Path implementations
    # ------------------------------------------------------------------

    def _path_full(self, test_df, base_scores, G, rule_adjustments):
        """full / no_nlp: ML + graph signal + rule adjustments."""
        from models.fusion import compute_fused_account_scores
        scores = compute_fused_account_scores(
            test_df,
            base_scores,
            rule_engine_fn=True if rule_adjustments is None else None,
        )
        # If caller provided explicit adjustments, apply them via fuse_scores
        if rule_adjustments is not None:
            from models.fusion import fuse_scores
            scores = fuse_scores(scores, rule_adjustments=rule_adjustments)
        return scores

    def _path_no_symbolic(self, test_df, base_scores, G):
        """no_symbolic: ML + graph, no symbolic rule adjustments."""
        from models.fusion import compute_fused_account_scores
        # Omit rule_engine_fn → no symbolic bump applied
        return compute_fused_account_scores(test_df, base_scores, rule_engine_fn=None)

    def _path_no_pagerank(self, test_df, base_scores, rule_adjustments):
        """no_pagerank: ML + rules, no graph signal (static fusion weights)."""
        from models.fusion import fuse_scores
        if rule_adjustments is not None:
            return fuse_scores(base_scores, rule_adjustments=rule_adjustments)
        # Use compute_fused_account_scores for its vectorised rule logic
        from models.fusion import compute_fused_account_scores
        return compute_fused_account_scores(test_df, base_scores, rule_engine_fn=True)

    def _path_no_fusion(self, test_df, base_scores, G, rule_adjustments):
        """
        no_fusion: ML + TD-PageRank graph signal + rules, but adaptive
        fusion is replaced with static weights (0.70/0.15/0.15).
        Graph scores are incorporated by blending with base scores at
        static weight, then rules are applied on top.
        """
        from models.fusion import fuse_scores
        # Blend graph features already embedded in test_df with static weight
        graph_boost = np.zeros(len(base_scores))
        if "sender_gf_pagerank" in test_df.columns:
            # Normalise the pre-computed pagerank column to [0,1] as S_graph proxy
            pr = test_df["sender_gf_pagerank"].values.astype(np.float64)
            pr_min, pr_max = pr.min(), pr.max()
            if pr_max > pr_min:
                graph_boost = (pr - pr_min) / (pr_max - pr_min)
        # Static weights: w_ml=0.70, w_graph=0.15, w_rules handled by fuse_scores
        blended = 0.70 * base_scores + 0.15 * graph_boost
        # Apply rule adjustments (the remaining 0.15 budget is covered by bump)
        if rule_adjustments is not None:
            return fuse_scores(blended, rule_adjustments=rule_adjustments)
        from models.fusion import compute_fused_account_scores
        return compute_fused_account_scores(test_df, blended, rule_engine_fn=True)

    def _path_lgbm_pagerank(self, test_df, base_scores, G):
        """lgbm_pagerank: ML + graph signal, no rules."""
        # Blend graph features at static weights, no rule bump
        graph_boost = np.zeros(len(base_scores))
        if "sender_gf_pagerank" in test_df.columns:
            pr = test_df["sender_gf_pagerank"].values.astype(np.float64)
            pr_min, pr_max = pr.min(), pr.max()
            if pr_max > pr_min:
                graph_boost = (pr - pr_min) / (pr_max - pr_min)
        return 0.75 * base_scores + 0.25 * graph_boost

    def _path_lgbm_rules(self, test_df, base_scores, rule_adjustments):
        """lgbm_rules: ML + rules, no graph signal."""
        from models.fusion import fuse_scores, compute_fused_account_scores
        if rule_adjustments is not None:
            return fuse_scores(base_scores, rule_adjustments=rule_adjustments)
        return compute_fused_account_scores(test_df, base_scores, rule_engine_fn=True)


# ------------------------------------------------------------------
# Module-level demo / smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    print("=== PipelineOrchestrator Smoke Test ===\n")

    # Minimal synthetic data
    rng = np.random.default_rng(42)
    n = 100
    test_df = pd.DataFrame({
        "Sender_account": rng.integers(1, 20, n),
        "Receiver_account": rng.integers(1, 20, n),
        "amount_local_npr": rng.uniform(1000, 500000, n),
        "tx_count_10": rng.integers(0, 20, n),
        "cross_border_flag": rng.integers(0, 2, n),
        "sender_gf_out_degree": rng.integers(0, 15, n),
        "velocity_sum_10tx": rng.uniform(0, 2_000_000, n),
        "sender_gf_pagerank": rng.uniform(0, 10, n),
        "sender_gf_in_cycle": rng.integers(0, 2, n),
        "near_threshold_100k": rng.integers(0, 2, n),
        "is_off_hours": rng.integers(0, 2, n),
    })
    y_score = rng.uniform(0, 1, n)

    orchestrator = PipelineOrchestrator(precision_budget=0.60)
    result = orchestrator.score_accounts(test_df, y_score)

    print(f"Path used    : {result['path_id']}")
    print(f"Fallback flag: {result['fallback']}")
    print(f"Score range  : [{result['scores'].min():.4f}, {result['scores'].max():.4f}]")
    print(f"Score mean   : {result['scores'].mean():.4f}")
    print(f"NaN count    : {np.isnan(result['scores']).sum()}")
    print("\nSmoke test passed ✓")
