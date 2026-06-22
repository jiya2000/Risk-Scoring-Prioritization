"""
Property 1: Bug Condition — Fabricated Prior Art Reference.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7

This exploration test checks that all PageRank prior art references use
"US11640609B1" (Wells Fargo, 2023). On UNFIXED code these tests
will FAIL because the code currently returns "US20240062041" — the failure
confirms the bug exists.

DO NOT fix the code to make these tests pass. The failure IS the expected
outcome for exploration.
"""

import sys
import logging
from pathlib import Path
from datetime import date

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Path setup (mirrors project convention)
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.data_models import PageRankNoveltyReport, PatentMetricsReport
from evaluation.patent_harness import PatentEvaluationHarness


# ---------------------------------------------------------------------------
# Hypothesis strategies (reused from test_patent_harness.py patterns)
# ---------------------------------------------------------------------------

@st.composite
def pagerank_score_pairs(draw):
    """
    Generate (td_scores, standard_scores) dicts for 3–50 nodes.
    Scores are raw floats in (0, 1].
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


# ---------------------------------------------------------------------------
# Bug Condition Exploration Tests
# ---------------------------------------------------------------------------

class TestBugConditionExploration:
    """
    Property 1: Bug Condition — Fabricated Prior Art Reference.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**

    These tests MUST FAIL on unfixed code to confirm the bug exists.
    """

    def test_pagerank_novelty_report_default_prior_art_ref(self):
        """
        Bug Condition: PageRankNoveltyReport().prior_art_ref should equal
        "US11640609B1" (Wells Fargo, 2023).
        """
        report = PageRankNoveltyReport(
            mean_absolute_difference=0.05,
            mean_absolute_pct_difference=0.10,
            meets_novelty_threshold=True,
        )
        assert report.prior_art_ref == "US11640609B1", (
            f"Expected prior_art_ref='US11640609B1', "
            f"got '{report.prior_art_ref}'"
        )

    @given(pr_pair=pagerank_score_pairs())
    @settings(max_examples=50, deadline=2000)
    def test_evaluate_td_pagerank_returns_correct_prior_art_ref(self, pr_pair):
        """
        Bug Condition: For any random TD-PageRank score pairs,
        evaluate_td_pagerank() should produce a report with
        prior_art_ref == "US11640609B1".
        """
        td_scores, standard_scores = pr_pair

        harness = PatentEvaluationHarness(random_state=42)
        harness.dataset_size = 100
        harness.temporal_split_date = date(2023, 8, 1)

        report = harness.evaluate_td_pagerank(td_scores, standard_scores)

        assert report.prior_art_ref == "US11640609B1", (
            f"Expected evaluate_td_pagerank() to return prior_art_ref='US11640609B1', "
            f"got '{report.prior_art_ref}'"
        )

    def test_format_report_contains_correct_prior_art_ref(self):
        """
        Bug Condition: _format_report() output should contain "US11640609B1"
        in the PageRank section header.
        """
        harness = PatentEvaluationHarness(random_state=42)
        harness.dataset_size = 100
        harness.temporal_split_date = date(2023, 8, 1)

        # Run all evaluations to produce a full report
        rng = np.random.default_rng(42)
        n = 100
        y_true = (rng.random(n) < 0.10).astype(int)
        adaptive_scores = rng.random(n)
        static_scores = rng.random(n)

        harness.evaluate_topology_adaptive_fusion(adaptive_scores, static_scores, y_true)
        harness.evaluate_td_pagerank(
            {f"n{i}": 0.05 + i * 0.001 for i in range(20)},
            {f"n{i}": 0.03 for i in range(20)},
        )
        harness.evaluate_degradation_paths({"full": 0.80, "lgbm_only": 0.62})

        report = harness.generate_report()
        formatted = harness._format_report(report)

        assert "US11640609B1" in formatted, (
            f"Expected _format_report() to contain 'US11640609B1' in PageRank section, "
            f"but it was not found in formatted output"
        )

    def test_evaluate_td_pagerank_logger_warning_references_correct_ref(self, caplog):
        """
        Bug Condition: The evaluate_td_pagerank logger warning (when threshold
        not met) should reference "US11640609B1".
        """
        harness = PatentEvaluationHarness(random_state=42)
        harness.dataset_size = 100
        harness.temporal_split_date = date(2023, 8, 1)

        # Use nearly identical scores so MAD < 0.01 → triggers warning
        nodes = [f"ACC_{i}" for i in range(10)]
        td_scores = {n: 0.010 for n in nodes}
        standard_scores = {n: 0.010 + 1e-5 for n in nodes}

        with caplog.at_level(logging.WARNING, logger="evaluation.patent_harness"):
            harness.evaluate_td_pagerank(td_scores, standard_scores)

        # Find the warning message about threshold not met
        warning_messages = [
            record.message for record in caplog.records
            if "novelty threshold NOT met" in record.message
            and "PageRank" in record.message
        ]

        assert len(warning_messages) > 0, (
            "Expected a logger warning about TD-PageRank novelty threshold not met"
        )

        for msg in warning_messages:
            assert "US11640609B1" in msg, (
                f"Expected logger warning to reference 'US11640609B1', "
                f"but got: '{msg}'"
            )
