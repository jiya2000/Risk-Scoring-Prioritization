"""
Patent Evaluation Harness for AML Innovations.

Measures each innovation's improvement over prior art baselines:
  - Topology-Adaptive Fusion vs. static fusion (prior art: US20220405860)
  - TD-PageRank vs. standard PageRank     (prior art: US20240062041)
  - Degradation Controller path precision  (prior art: US20260038036)

Produces structured PatentMetricsReport suitable for patent claims.
"""

import sys
import os
import logging
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import (
    FusionNoveltyReport,
    PageRankNoveltyReport,
    DegradationNoveltyReport,
    PatentMetricsReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precision@K helper
# ---------------------------------------------------------------------------

def _precision_at_k(scores: np.ndarray, y_true: np.ndarray, k: int = 50) -> float:
    """
    Return Precision@k: fraction of true positives in the top-k ranked items.

    Args:
        scores: Array of predicted risk scores (higher = more suspicious).
        y_true: Binary ground-truth labels (1 = suspicious, 0 = benign).
        k:      Number of top-ranked items to evaluate.

    Returns:
        Precision@k as a float in [0.0, 1.0].
    """
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    top_k_indices = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[top_k_indices]))


# ---------------------------------------------------------------------------
# PatentEvaluationHarness
# ---------------------------------------------------------------------------

class PatentEvaluationHarness:
    """
    Measures each AML innovation's improvement over prior-art baselines.

    Usage:
        harness = PatentEvaluationHarness(random_state=42)
        harness.evaluate_topology_adaptive_fusion(adaptive, static, y_true)
        harness.evaluate_td_pagerank(td_scores, std_scores)
        harness.evaluate_degradation_paths(path_results)
        report = harness.generate_report()
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

        self._fusion_report: Optional[FusionNoveltyReport] = None
        self._pagerank_report: Optional[PageRankNoveltyReport] = None
        self._degradation_report: Optional[DegradationNoveltyReport] = None

        # Metadata populated by the caller or derived from data
        self.dataset_size: int = 0
        self.temporal_split_date: date = date.today()

    # ------------------------------------------------------------------
    # Innovation 1 – Topology-Adaptive Fusion vs. static fusion
    # ------------------------------------------------------------------

    def evaluate_topology_adaptive_fusion(
        self,
        adaptive_scores: np.ndarray,
        static_scores: np.ndarray,
        y_true: np.ndarray,
        k: int = 50,
    ) -> FusionNoveltyReport:
        """
        Compute Precision@50 for adaptive and static fusion, then derive the
        relative improvement percentage and novelty threshold flag.

        Args:
            adaptive_scores: Predicted risk scores from Topology-Adaptive Fusion.
            static_scores:   Predicted risk scores from the static sigmoid baseline.
            y_true:          Binary ground-truth labels (1 = suspicious).
            k:               Rank cut-off (default 50).

        Returns:
            FusionNoveltyReport with improvement metrics and flag.
        """
        adaptive_scores = np.asarray(adaptive_scores, dtype=float)
        static_scores = np.asarray(static_scores, dtype=float)
        y_true = np.asarray(y_true, dtype=int)

        if len(adaptive_scores) != len(y_true) or len(static_scores) != len(y_true):
            raise ValueError(
                "adaptive_scores, static_scores, and y_true must have the same length."
            )

        adaptive_p50 = _precision_at_k(adaptive_scores, y_true, k)
        static_p50 = _precision_at_k(static_scores, y_true, k)

        absolute_improvement = adaptive_p50 - static_p50

        # Avoid division by zero when static_p50 == 0
        if static_p50 > 0:
            relative_improvement_pct = (adaptive_p50 - static_p50) / static_p50 * 100.0
        else:
            # If static precision is 0 and adaptive is > 0, treat as infinite improvement
            relative_improvement_pct = float("inf") if adaptive_p50 > 0 else 0.0

        meets_novelty_threshold = relative_improvement_pct >= 10.0

        if not meets_novelty_threshold:
            logger.warning(
                "Fusion novelty threshold NOT met: relative_improvement_pct=%.2f%% "
                "(required >= 10%%). Flagging component as insufficiently differentiated "
                "from prior art US20220405860.",
                relative_improvement_pct,
            )

        self._fusion_report = FusionNoveltyReport(
            adaptive_precision_at_50=adaptive_p50,
            static_precision_at_50=static_p50,
            absolute_improvement=absolute_improvement,
            relative_improvement_pct=relative_improvement_pct,
            meets_novelty_threshold=meets_novelty_threshold,
            prior_art_ref="US20220405860",
        )

        # Update dataset_size based on the evaluation set size
        self.dataset_size = len(y_true)

        return self._fusion_report

    # ------------------------------------------------------------------
    # Innovation 2 – TD-PageRank vs. standard PageRank
    # ------------------------------------------------------------------

    def evaluate_td_pagerank(
        self,
        td_scores: Dict[str, float],
        standard_scores: Dict[str, float],
    ) -> PageRankNoveltyReport:
        """
        Compute the mean absolute difference between TD-PageRank and standard
        PageRank scores across all common nodes.

        Args:
            td_scores:       Node → TD-PageRank score mapping.
            standard_scores: Node → standard PageRank score mapping.

        Returns:
            PageRankNoveltyReport with differentiation metrics and flag.
        """
        if not td_scores or not standard_scores:
            raise ValueError("Both td_scores and standard_scores must be non-empty.")

        # Evaluate only on nodes present in both dicts
        common_nodes = set(td_scores.keys()) & set(standard_scores.keys())
        if not common_nodes:
            raise ValueError(
                "td_scores and standard_scores share no common nodes; "
                "cannot compute mean absolute difference."
            )

        td_arr = np.array([td_scores[n] for n in common_nodes], dtype=float)
        std_arr = np.array([standard_scores[n] for n in common_nodes], dtype=float)

        abs_diff = np.abs(td_arr - std_arr)
        mean_absolute_difference = float(np.mean(abs_diff))

        # Mean absolute percentage difference – guard against std = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_diff = np.where(std_arr != 0, abs_diff / std_arr, 0.0)
        mean_absolute_pct_difference = float(np.mean(pct_diff))

        meets_novelty_threshold = mean_absolute_difference >= 0.01

        if not meets_novelty_threshold:
            logger.warning(
                "TD-PageRank novelty threshold NOT met: mean_absolute_difference=%.6f "
                "(required >= 0.01). Flagging TD-PageRank as insufficiently "
                "differentiated from prior art US20240062041. "
                "Observed difference: %.6f",
                mean_absolute_difference,
                mean_absolute_difference,
            )

        self._pagerank_report = PageRankNoveltyReport(
            mean_absolute_difference=mean_absolute_difference,
            mean_absolute_pct_difference=mean_absolute_pct_difference,
            meets_novelty_threshold=meets_novelty_threshold,
            prior_art_ref="US20240062041",
        )

        return self._pagerank_report

    # ------------------------------------------------------------------
    # Innovation 3 – Degradation Controller path precision
    # ------------------------------------------------------------------

    def evaluate_degradation_paths(
        self,
        path_results: Dict[str, float],
    ) -> DegradationNoveltyReport:
        """
        Verify that each execution path meets the Precision@50 budget (>= 0.60)
        and derive the minimum precision maintained across all paths.

        Args:
            path_results: Mapping of path_id → measured Precision@50.

        Returns:
            DegradationNoveltyReport with path-level precision and budget flag.
        """
        if not path_results:
            raise ValueError("path_results must contain at least one entry.")

        precision_budget = 0.60

        min_precision_maintained = float(min(path_results.values()))
        meets_precision_budget = all(
            p >= precision_budget for p in path_results.values()
        )

        if not meets_precision_budget:
            failing_paths = {
                pid: p for pid, p in path_results.items() if p < precision_budget
            }
            logger.warning(
                "Degradation precision budget NOT met for paths: %s "
                "(required each >= %.2f). Prior art: US20260038036.",
                failing_paths,
                precision_budget,
            )

        self._degradation_report = DegradationNoveltyReport(
            path_precisions=dict(path_results),
            min_precision_maintained=min_precision_maintained,
            meets_precision_budget=meets_precision_budget,
            prior_art_ref="US20260038036",
        )

        return self._degradation_report

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> PatentMetricsReport:
        """
        Produce a structured PatentMetricsReport containing all three
        sub-reports plus metadata (dataset size, split date, random seed).

        Returns:
            PatentMetricsReport with labeled sections per prior art reference.

        Raises:
            RuntimeError: If any evaluate_* method has not been called yet.
        """
        missing = []
        if self._fusion_report is None:
            missing.append("evaluate_topology_adaptive_fusion()")
        if self._pagerank_report is None:
            missing.append("evaluate_td_pagerank()")
        if self._degradation_report is None:
            missing.append("evaluate_degradation_paths()")

        if missing:
            raise RuntimeError(
                f"Cannot generate report: the following evaluations have not been "
                f"completed yet: {', '.join(missing)}"
            )

        report = PatentMetricsReport(
            fusion_report=self._fusion_report,
            pagerank_report=self._pagerank_report,
            degradation_report=self._degradation_report,
            dataset_size=self.dataset_size,
            temporal_split_date=self.temporal_split_date,
            random_seed=self.random_state,
            generated_at=datetime.now(),
        )

        return report

    # ------------------------------------------------------------------
    # Pretty-print helper
    # ------------------------------------------------------------------

    def _format_report(self, report: PatentMetricsReport) -> str:
        """Return a human-readable string representation of the report."""
        lines = [
            "=" * 70,
            "  PATENT METRICS REPORT",
            f"  Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Dataset size:        {report.dataset_size}",
            f"  Temporal split date: {report.temporal_split_date}",
            f"  Random seed:         {report.random_seed}",
            "=" * 70,
            "",
            "── Prior Art: US20220405860 (Static Weighted Ensemble) ─────────────",
            f"  Adaptive Fusion  Precision@50:  {report.fusion_report.adaptive_precision_at_50:.4f}",
            f"  Static Fusion    Precision@50:  {report.fusion_report.static_precision_at_50:.4f}",
            f"  Absolute Improvement:           {report.fusion_report.absolute_improvement:+.4f}",
            f"  Relative Improvement:           {report.fusion_report.relative_improvement_pct:.2f}%",
            f"  Meets Novelty Threshold (≥10%): {'YES ✓' if report.fusion_report.meets_novelty_threshold else 'NO ✗  [FLAGGED]'}",
            "",
            "── Prior Art: US20240062041 (Standard Centrality Metrics) ──────────",
            f"  Mean Absolute Difference:       {report.pagerank_report.mean_absolute_difference:.6f}",
            f"  Mean Absolute Pct Difference:   {report.pagerank_report.mean_absolute_pct_difference:.4f}",
            f"  Meets Novelty Threshold (≥0.01):{' YES ✓' if report.pagerank_report.meets_novelty_threshold else ' NO ✗  [FLAGGED]'}",
            "",
            "── Prior Art: US20260038036 (ML Pipeline Monitoring) ───────────────",
            f"  Min Precision Maintained:       {report.degradation_report.min_precision_maintained:.4f}",
            f"  Meets Precision Budget (≥0.60): {'YES ✓' if report.degradation_report.meets_precision_budget else 'NO ✗  [FLAGGED]'}",
            "  Path-level Precisions:",
        ]
        for pid, p in sorted(report.degradation_report.path_precisions.items()):
            flag = "" if p >= 0.60 else "  ← BELOW BUDGET"
            lines.append(f"    {pid:<22} {p:.4f}{flag}")
        lines += ["", "=" * 70]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# __main__ demo with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("PatentEvaluationHarness – synthetic data demo")
    print("-" * 50)

    rng = np.random.default_rng(42)
    N = 500  # evaluation set size

    # ── Synthetic ground truth (10% suspicious rate) ─────────────────────
    y_true = (rng.random(N) < 0.10).astype(int)

    # ── Adaptive fusion scores (slightly better, more suspicious → higher) ─
    adaptive_scores = rng.random(N)
    adaptive_scores[y_true == 1] += 0.25          # boost true positives
    adaptive_scores = np.clip(adaptive_scores, 0, 1)

    # ── Static fusion scores (baseline – slightly weaker) ─────────────────
    static_scores = rng.random(N)
    static_scores[y_true == 1] += 0.18            # smaller boost
    static_scores = np.clip(static_scores, 0, 1)

    # ── TD-PageRank vs standard PageRank scores for 100 nodes ─────────────
    nodes = [f"ACC_{i}" for i in range(100)]
    # Standard PageRank (uniform-ish, sums to ~1 after normalisation)
    std_raw = rng.random(100) + 0.01
    std_raw /= std_raw.sum()
    standard_pr = dict(zip(nodes, std_raw.tolist()))

    # TD-PageRank – apply decay + cycle penalty → shift distribution
    td_raw = std_raw * rng.uniform(0.5, 1.5, size=100)
    td_raw /= td_raw.sum()
    td_pr = dict(zip(nodes, td_raw.tolist()))

    # ── Degradation path precision table (all paths above budget) ─────────
    path_results = {
        "full":         0.82,
        "no_nlp":       0.82,
        "no_symbolic":  0.76,
        "no_pagerank":  0.72,
        "no_fusion":    0.70,
        "lgbm_pagerank": 0.68,
        "lgbm_rules":   0.66,
        "lgbm_only":    0.62,
    }

    # ── Run harness ────────────────────────────────────────────────────────
    harness = PatentEvaluationHarness(random_state=42)
    harness.dataset_size = N
    harness.temporal_split_date = date(2023, 8, 1)

    fusion_rep = harness.evaluate_topology_adaptive_fusion(
        adaptive_scores, static_scores, y_true
    )
    pr_rep = harness.evaluate_td_pagerank(td_pr, standard_pr)
    deg_rep = harness.evaluate_degradation_paths(path_results)
    report = harness.generate_report()

    print(harness._format_report(report))
