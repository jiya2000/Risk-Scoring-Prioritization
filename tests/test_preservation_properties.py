"""
Property 2: Preservation — Non-PageRank Prior Art References and Computation Logic.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

These tests observe and preserve CURRENT CORRECT behavior that must NOT change
after the bugfix. They verify:
  - Fusion prior art reference always "US20220405860"
  - Degradation prior art reference always "US20210174258"
  - TD-PageRank novelty threshold logic (meets_novelty_threshold iff MAD >= 0.01)
  - Report structure contains all three sub-reports with correct field types
"""

import sys
from pathlib import Path
from datetime import date
from typing import Dict

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Path setup (mirrors project convention of using sys.path.append)
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.patent_harness import PatentEvaluationHarness
from models.data_models import (
    FusionNoveltyReport,
    PageRankNoveltyReport,
    DegradationNoveltyReport,
    PatentMetricsReport,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies (mirroring test_patent_harness.py)
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
# Helper to build a fully-evaluated harness
# ---------------------------------------------------------------------------

def _make_full_harness(
    adaptive_scores,
    static_scores,
    y_true_list,
    td_scores: Dict[str, float],
    standard_scores: Dict[str, float],
    path_results: Dict[str, float],
) -> PatentEvaluationHarness:
    """Create a harness with all three evaluations completed."""
    h = PatentEvaluationHarness(random_state=42)
    h.dataset_size = len(y_true_list)
    h.temporal_split_date = date(2023, 8, 1)
    h.evaluate_topology_adaptive_fusion(
        np.asarray(adaptive_scores),
        np.asarray(static_scores),
        np.asarray(y_true_list),
    )
    h.evaluate_td_pagerank(td_scores, standard_scores)
    h.evaluate_degradation_paths(path_results)
    return h


# ---------------------------------------------------------------------------
# Preservation Property Tests
# ---------------------------------------------------------------------------

class TestPreservationProperties:
    """
    Property 2: Preservation — Non-PageRank Prior Art and Computation Logic.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

    These tests verify behavior that MUST remain unchanged after the bugfix.
    All tests MUST PASS on current (unfixed) code.
    """

    @given(fusion_inputs=fusion_evaluation_inputs())
    @settings(max_examples=100, deadline=2000)
    def test_fusion_prior_art_ref_preserved(self, fusion_inputs):
        """
        **Validates: Requirements 3.1**

        For all random fusion evaluation inputs:
        report.fusion_report.prior_art_ref == "US20220405860" always holds.
        """
        adaptive, static_, y_true = fusion_inputs

        harness = PatentEvaluationHarness(random_state=42)
        harness.dataset_size = len(y_true)
        harness.temporal_split_date = date(2023, 8, 1)

        report = harness.evaluate_topology_adaptive_fusion(
            np.asarray(adaptive),
            np.asarray(static_),
            np.asarray(y_true),
        )

        assert report.prior_art_ref == "US20220405860", (
            f"Fusion prior_art_ref should always be 'US20220405860', "
            f"got '{report.prior_art_ref}'"
        )

    @given(deg_paths=degradation_path_results())
    @settings(max_examples=100, deadline=2000)
    def test_degradation_prior_art_ref_preserved(self, deg_paths):
        """
        **Validates: Requirements 3.2**

        For all random degradation path results:
        report.degradation_report.prior_art_ref == "US20210174258" always holds.
        """
        harness = PatentEvaluationHarness(random_state=42)
        harness.dataset_size = 100
        harness.temporal_split_date = date(2023, 8, 1)

        report = harness.evaluate_degradation_paths(deg_paths)

        assert report.prior_art_ref == "US20210174258", (
            f"Degradation prior_art_ref should always be 'US20210174258', "
            f"got '{report.prior_art_ref}'"
        )

    @given(pr_pair=pagerank_score_pairs())
    @settings(max_examples=100, deadline=2000)
    def test_pagerank_threshold_logic_preserved(self, pr_pair):
        """
        **Validates: Requirements 3.3, 3.5**

        For all random TD-PageRank score pairs:
        report.pagerank_report.meets_novelty_threshold ==
            (report.pagerank_report.mean_absolute_difference >= 0.01)
        — mathematical threshold is preserved.
        """
        td_scores, std_scores = pr_pair

        harness = PatentEvaluationHarness(random_state=42)
        harness.dataset_size = 100
        harness.temporal_split_date = date(2023, 8, 1)

        report = harness.evaluate_td_pagerank(td_scores, std_scores)

        expected_flag = report.mean_absolute_difference >= 0.01
        assert report.meets_novelty_threshold == expected_flag, (
            f"TD-PageRank threshold logic violated: "
            f"mean_absolute_difference={report.mean_absolute_difference:.6f}, "
            f"meets_novelty_threshold={report.meets_novelty_threshold}, "
            f"expected={expected_flag}"
        )

    @given(
        fusion_inputs=fusion_evaluation_inputs(),
        pr_pair=pagerank_score_pairs(),
        deg_paths=degradation_path_results(),
    )
    @settings(max_examples=100, deadline=2000)
    def test_report_structure_preserved(self, fusion_inputs, pr_pair, deg_paths):
        """
        **Validates: Requirements 3.4**

        For all complete evaluation triples: report contains
        fusion_report (FusionNoveltyReport), pagerank_report (PageRankNoveltyReport),
        degradation_report (DegradationNoveltyReport), and all boolean flags
        are proper booleans.
        """
        adaptive, static_, y_true = fusion_inputs
        td_scores, std_scores = pr_pair

        harness = _make_full_harness(
            adaptive, static_, y_true, td_scores, std_scores, deg_paths
        )
        report = harness.generate_report()

        # Verify structure: all three sub-reports present with correct types
        assert isinstance(report, PatentMetricsReport), (
            f"Report should be PatentMetricsReport, got {type(report)}"
        )
        assert isinstance(report.fusion_report, FusionNoveltyReport), (
            f"fusion_report should be FusionNoveltyReport, got {type(report.fusion_report)}"
        )
        assert isinstance(report.pagerank_report, PageRankNoveltyReport), (
            f"pagerank_report should be PageRankNoveltyReport, got {type(report.pagerank_report)}"
        )
        assert isinstance(report.degradation_report, DegradationNoveltyReport), (
            f"degradation_report should be DegradationNoveltyReport, got {type(report.degradation_report)}"
        )

        # Verify all boolean flags are proper booleans
        assert isinstance(report.fusion_report.meets_novelty_threshold, bool), (
            f"fusion meets_novelty_threshold should be bool, "
            f"got {type(report.fusion_report.meets_novelty_threshold)}"
        )
        assert isinstance(report.pagerank_report.meets_novelty_threshold, bool), (
            f"pagerank meets_novelty_threshold should be bool, "
            f"got {type(report.pagerank_report.meets_novelty_threshold)}"
        )
        assert isinstance(report.degradation_report.meets_precision_budget, bool), (
            f"degradation meets_precision_budget should be bool, "
            f"got {type(report.degradation_report.meets_precision_budget)}"
        )
