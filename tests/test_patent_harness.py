"""
Property 15: Patent metrics report structure and threshold flagging.

Validates: Requirements 4.5, 4.6, 4.7

For any set of evaluation results (fusion improvement, PageRank difference,
path precisions), the PatentEvaluationHarness SHALL:
  - Produce a report containing all required fields
    (numeric improvement, dataset size, temporal split date, random seed,
    prior art sections).
  - Flag the fusion component when relative_improvement_pct < 10%.
  - Flag the TD-PageRank component when mean_absolute_difference < 0.01.
"""

import sys
from pathlib import Path
from datetime import date
from typing import Dict

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Path setup (mirrors project convention of using sys.path.append)
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.patent_harness import PatentEvaluationHarness, _precision_at_k
from models.data_models import (
    FusionNoveltyReport,
    PageRankNoveltyReport,
    DegradationNoveltyReport,
    PatentMetricsReport,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def fusion_evaluation_inputs(draw):
    """
    Generate (adaptive_scores, static_scores, y_true) triples.

    Produces N items (10–200) with binary labels (5–30% positive rate) and
    independent score vectors in [0, 1].
    """
    n = draw(st.integers(min_value=10, max_value=200))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    pos_rate = draw(st.floats(min_value=0.05, max_value=0.30))
    y_true = (rng.random(n) < pos_rate).astype(int)

    adaptive_scores = rng.random(n).tolist()
    static_scores = rng.random(n).tolist()

    return adaptive_scores, static_scores, y_true.tolist()


@st.composite
def pagerank_score_pairs(draw):
    """
    Generate (td_scores, standard_scores) dicts for 3–50 nodes.

    Scores are raw floats in (0, 1]; they need not sum to 1 because the
    harness computes differences directly without normalisation.
    """
    n = draw(st.integers(min_value=3, max_value=50))
    nodes = [f"ACC_{i}" for i in range(n)]

    td_scores = {
        node: draw(st.floats(min_value=1e-6, max_value=1.0,
                             allow_nan=False, allow_infinity=False))
        for node in nodes
    }
    standard_scores = {
        node: draw(st.floats(min_value=1e-6, max_value=1.0,
                             allow_nan=False, allow_infinity=False))
        for node in nodes
    }
    return td_scores, standard_scores


@st.composite
def degradation_path_results(draw):
    """
    Generate a dict of path_id → precision for 1–8 paths.
    Precisions are in [0.0, 1.0].
    """
    n_paths = draw(st.integers(min_value=1, max_value=8))
    path_ids = [f"path_{i}" for i in range(n_paths)]
    return {
        pid: draw(st.floats(min_value=0.0, max_value=1.0,
                            allow_nan=False, allow_infinity=False))
        for pid in path_ids
    }


# ---------------------------------------------------------------------------
# Unit tests (deterministic examples)
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    """Unit tests for the _precision_at_k helper."""

    def test_perfect_ranking(self):
        """Top-k items are all positives → precision = 1.0."""
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.05])
        y_true = np.array([1, 1, 1, 0, 0])
        assert _precision_at_k(scores, y_true, k=3) == 1.0

    def test_zero_precision(self):
        """Top-k items are all negatives → precision = 0.0."""
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.05])
        y_true = np.array([0, 0, 0, 1, 1])
        assert _precision_at_k(scores, y_true, k=3) == 0.0

    def test_k_exceeds_length(self):
        """k larger than the array → clipped to len(scores), no error."""
        scores = np.array([0.9, 0.5])
        y_true = np.array([1, 0])
        result = _precision_at_k(scores, y_true, k=100)
        assert 0.0 <= result <= 1.0

    def test_empty_input(self):
        """Empty arrays → returns 0.0."""
        assert _precision_at_k(np.array([]), np.array([]), k=50) == 0.0


class TestHarnessUnit:
    """Deterministic unit tests for PatentEvaluationHarness methods."""

    def _make_harness(self) -> PatentEvaluationHarness:
        h = PatentEvaluationHarness(random_state=42)
        h.dataset_size = 100
        h.temporal_split_date = date(2023, 8, 1)
        return h

    def test_fusion_above_threshold(self):
        """Relative improvement >= 10% → meets_novelty_threshold = True."""
        harness = self._make_harness()
        # adaptive is clearly better in top-50
        np.random.seed(0)
        n = 200
        y_true = np.zeros(n, dtype=int)
        y_true[:20] = 1  # 20 positives

        # Static scores: positives ranked ~50th–70th
        static = np.linspace(0.0, 1.0, n)
        np.random.shuffle(static)

        # Adaptive scores: positives pushed firmly into top-50
        adaptive = static.copy()
        pos_idx = np.where(y_true == 1)[0]
        adaptive[pos_idx] = 0.95 + np.random.rand(len(pos_idx)) * 0.05

        rep = harness.evaluate_topology_adaptive_fusion(adaptive, static, y_true)
        assert rep.meets_novelty_threshold is True
        assert rep.prior_art_ref == "US20220405860"

    def test_fusion_below_threshold_flagged(self):
        """Relative improvement < 10% → meets_novelty_threshold = False."""
        harness = self._make_harness()
        # Same scores → 0% improvement
        scores = np.random.rand(200)
        y_true = (np.random.rand(200) > 0.8).astype(int)

        rep = harness.evaluate_topology_adaptive_fusion(scores, scores, y_true)
        assert rep.meets_novelty_threshold is False
        assert rep.relative_improvement_pct == 0.0

    def test_pagerank_above_threshold(self):
        """Mean absolute difference >= 0.01 → meets_novelty_threshold = True."""
        harness = self._make_harness()
        nodes = [f"ACC_{i}" for i in range(50)]
        td = {n: 0.05 for n in nodes}
        std = {n: 0.03 for n in nodes}   # MAD = 0.02 > 0.01

        rep = harness.evaluate_td_pagerank(td, std)
        assert rep.meets_novelty_threshold is True
        assert abs(rep.mean_absolute_difference - 0.02) < 1e-9

    def test_pagerank_below_threshold_flagged(self):
        """Mean absolute difference < 0.01 → meets_novelty_threshold = False."""
        harness = self._make_harness()
        nodes = [f"ACC_{i}" for i in range(10)]
        td = {n: 0.010 for n in nodes}
        std = {n: 0.010 + 1e-5 for n in nodes}   # MAD ≈ 1e-5 < 0.01

        rep = harness.evaluate_td_pagerank(td, std)
        assert rep.meets_novelty_threshold is False

    def test_degradation_budget_met(self):
        """All paths >= 0.60 → meets_precision_budget = True."""
        harness = self._make_harness()
        path_results = {"full": 0.82, "lgbm_only": 0.62}
        rep = harness.evaluate_degradation_paths(path_results)
        assert rep.meets_precision_budget is True
        assert rep.min_precision_maintained == pytest.approx(0.62)

    def test_degradation_budget_not_met(self):
        """A path with precision 0.50 < 0.60 → meets_precision_budget = False."""
        harness = self._make_harness()
        path_results = {"full": 0.82, "bad_path": 0.50}
        rep = harness.evaluate_degradation_paths(path_results)
        assert rep.meets_precision_budget is False
        assert rep.min_precision_maintained == pytest.approx(0.50)

    def test_generate_report_requires_all_evaluations(self):
        """generate_report() raises if any evaluate_* not called."""
        harness = self._make_harness()
        with pytest.raises(RuntimeError, match="evaluations have not been completed"):
            harness.generate_report()

    def test_full_report_structure(self):
        """generate_report() returns PatentMetricsReport with all required fields."""
        harness = self._make_harness()
        y_true = np.array([1, 0, 1, 0, 1] * 20)
        scores = np.random.rand(100)

        harness.evaluate_topology_adaptive_fusion(scores, scores * 0.9, y_true)
        harness.evaluate_td_pagerank(
            {f"n{i}": 0.05 + i * 0.001 for i in range(20)},
            {f"n{i}": 0.03 for i in range(20)},
        )
        harness.evaluate_degradation_paths({"full": 0.80, "lgbm_only": 0.62})

        report = harness.generate_report()

        assert isinstance(report, PatentMetricsReport)
        assert isinstance(report.fusion_report, FusionNoveltyReport)
        assert isinstance(report.pagerank_report, PageRankNoveltyReport)
        assert isinstance(report.degradation_report, DegradationNoveltyReport)
        assert report.random_seed == 42
        assert report.dataset_size == 100
        assert report.temporal_split_date == date(2023, 8, 1)
        assert report.generated_at is not None


# ---------------------------------------------------------------------------
# Property 15: Patent metrics report structure and threshold flagging
# ---------------------------------------------------------------------------

class TestProperty15:
    """
    Property 15: Patent metrics report structure and threshold flagging.

    **Validates: Requirements 4.5, 4.6, 4.7**

    For any random evaluation metric tuples:
      1. The report contains all required fields.
      2. Fusion is flagged (meets_novelty_threshold=False)
         when relative_improvement_pct < 10%.
      3. TD-PageRank is flagged (meets_novelty_threshold=False)
         when mean_absolute_difference < 0.01.
    """

    @staticmethod
    def _make_full_harness(
        adaptive_scores,
        static_scores,
        y_true_list,
        td_scores: Dict[str, float],
        standard_scores: Dict[str, float],
        path_results: Dict[str, float],
        dataset_size: int = 100,
    ) -> PatentEvaluationHarness:
        h = PatentEvaluationHarness(random_state=42)
        h.dataset_size = dataset_size
        h.temporal_split_date = date(2023, 8, 1)
        h.evaluate_topology_adaptive_fusion(
            np.asarray(adaptive_scores),
            np.asarray(static_scores),
            np.asarray(y_true_list),
        )
        h.evaluate_td_pagerank(td_scores, standard_scores)
        h.evaluate_degradation_paths(path_results)
        return h

    @given(
        fusion_inputs=fusion_evaluation_inputs(),
        pr_pair=pagerank_score_pairs(),
        deg_paths=degradation_path_results(),
    )
    @settings(max_examples=100, deadline=2000)
    def test_report_contains_all_required_fields(
        self,
        fusion_inputs,
        pr_pair,
        deg_paths,
    ):
        """
        Property 15a: For any inputs, the generated report contains all
        required fields with correct types.
        """
        adaptive, static_, y_true = fusion_inputs
        td_scores, std_scores = pr_pair

        harness = self._make_full_harness(
            adaptive, static_, y_true, td_scores, std_scores, deg_paths
        )
        report = harness.generate_report()

        # ── Structural invariants ─────────────────────────────────────────
        assert isinstance(report, PatentMetricsReport), "report must be PatentMetricsReport"
        assert isinstance(report.fusion_report, FusionNoveltyReport)
        assert isinstance(report.pagerank_report, PageRankNoveltyReport)
        assert isinstance(report.degradation_report, DegradationNoveltyReport)

        # Numeric improvement fields
        assert isinstance(report.fusion_report.relative_improvement_pct, float)
        assert isinstance(report.fusion_report.absolute_improvement, float)
        assert isinstance(report.pagerank_report.mean_absolute_difference, float)

        # Metadata fields
        assert isinstance(report.dataset_size, int)
        assert isinstance(report.temporal_split_date, date)
        assert isinstance(report.random_seed, int)
        assert report.generated_at is not None

        # Prior art reference labels
        assert report.fusion_report.prior_art_ref == "US20220405860"
        assert report.pagerank_report.prior_art_ref == "US11640609B1"
        assert report.degradation_report.prior_art_ref == "US20210174258"

        # Boolean flags are proper booleans
        assert isinstance(report.fusion_report.meets_novelty_threshold, bool)
        assert isinstance(report.pagerank_report.meets_novelty_threshold, bool)
        assert isinstance(report.degradation_report.meets_precision_budget, bool)

    @given(
        fusion_inputs=fusion_evaluation_inputs(),
        pr_pair=pagerank_score_pairs(),
        deg_paths=degradation_path_results(),
    )
    @settings(max_examples=100, deadline=2000)
    def test_fusion_flagged_when_improvement_below_threshold(
        self,
        fusion_inputs,
        pr_pair,
        deg_paths,
    ):
        """
        Property 15b: Fusion is flagged (meets_novelty_threshold=False)
        iff relative_improvement_pct < 10%.
        """
        adaptive, static_, y_true = fusion_inputs
        td_scores, std_scores = pr_pair

        harness = self._make_full_harness(
            adaptive, static_, y_true, td_scores, std_scores, deg_paths
        )
        report = harness.generate_report()

        rel_imp = report.fusion_report.relative_improvement_pct
        flag = report.fusion_report.meets_novelty_threshold

        if rel_imp < 10.0:
            assert flag is False, (
                f"Fusion should be FLAGGED when relative_improvement_pct={rel_imp:.4f} < 10%, "
                f"but meets_novelty_threshold={flag}"
            )
        else:
            assert flag is True, (
                f"Fusion should NOT be flagged when relative_improvement_pct={rel_imp:.4f} >= 10%, "
                f"but meets_novelty_threshold={flag}"
            )

    @given(
        fusion_inputs=fusion_evaluation_inputs(),
        pr_pair=pagerank_score_pairs(),
        deg_paths=degradation_path_results(),
    )
    @settings(max_examples=100, deadline=2000)
    def test_pagerank_flagged_when_mad_below_threshold(
        self,
        fusion_inputs,
        pr_pair,
        deg_paths,
    ):
        """
        Property 15c: TD-PageRank is flagged (meets_novelty_threshold=False)
        iff mean_absolute_difference < 0.01.
        """
        adaptive, static_, y_true = fusion_inputs
        td_scores, std_scores = pr_pair

        harness = self._make_full_harness(
            adaptive, static_, y_true, td_scores, std_scores, deg_paths
        )
        report = harness.generate_report()

        mad = report.pagerank_report.mean_absolute_difference
        flag = report.pagerank_report.meets_novelty_threshold

        if mad < 0.01:
            assert flag is False, (
                f"TD-PageRank should be FLAGGED when mean_absolute_difference={mad:.6f} < 0.01, "
                f"but meets_novelty_threshold={flag}"
            )
        else:
            assert flag is True, (
                f"TD-PageRank should NOT be flagged when mean_absolute_difference={mad:.6f} >= 0.01, "
                f"but meets_novelty_threshold={flag}"
            )
