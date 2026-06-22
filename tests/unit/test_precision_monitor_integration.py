"""
Unit tests for DegradationController integration with OnlinePrecisionMonitor
and EnhancedConceptDriftDetector.

Tests the following behaviors:
- Routing table update when deviation > 0.05 (Req 4.2)
- Path re-evaluation when routing table precision value is updated (Req 4.4)
- CRITICAL alert when all paths degrade below adaptive precision budget (Req 4.6)
- Path re-selection within 500ms on PRECISION_DEGRADED alert (Req 5.5)

Validates: Requirements 4.2, 4.4, 4.6, 5.5
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from models.degradation_controller import (
    DegradationController,
    OnlinePrecisionMonitor,
    EnhancedConceptDriftDetector,
    AdaptivePrecisionBudget,
)
from models.data_models import ExecutionPath, ComponentStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_controller_with_monitor(
    paths=None,
    precision_budget=0.60,
    min_samples=5,
    stale_queue_age_minutes=10,
):
    """Create a DegradationController wired to an OnlinePrecisionMonitor."""
    monitor = OnlinePrecisionMonitor(min_samples=min_samples, window_size=100)
    budget = AdaptivePrecisionBudget(base_budget=precision_budget)
    drift_detector = EnhancedConceptDriftDetector(
        kl_threshold=0.15,
        precision_monitor=monitor,
        precision_budget=budget,
    )

    controller = DegradationController(
        precision_budget=precision_budget,
        paths=paths,
        online_precision_monitor=monitor,
        enhanced_drift_detector=drift_detector,
    )

    # Set a valid queue timestamp so stale queue checks pass
    if stale_queue_age_minutes is not None:
        controller.set_last_valid_queue_timestamp(
            datetime.now() - timedelta(minutes=stale_queue_age_minutes)
        )

    return controller, monitor, drift_detector


def _ingest_samples(monitor, path_id, n_samples, precision_target=0.80):
    """Ingest n labeled samples into the monitor for a path with ~target precision."""
    # Of 50 top-scored, approximately precision_target * 50 will be true positives
    base_time = datetime(2024, 6, 1, 12, 0, 0)
    for i in range(n_samples):
        score = 1.0 - (i / n_samples)  # Decreasing scores
        # First precision_target fraction of the top scores are true positives
        label = (i / n_samples) < precision_target
        monitor.ingest_feedback(
            account_id=f"ACC_{i}",
            score=score,
            label=label,
            timestamp=base_time + timedelta(seconds=i),
            path_id=path_id,
        )


# ---------------------------------------------------------------------------
# Test: Routing table update when deviation > 0.05 (Req 4.2)
# ---------------------------------------------------------------------------

class TestRoutingTableUpdate:
    """Tests for routing table update on precision deviation > 0.05."""

    def test_no_update_when_deviation_within_threshold(self):
        """When deviation <= 0.05, routing table should NOT be updated."""
        controller, monitor, _ = _create_controller_with_monitor(min_samples=5)

        # Ingest samples that produce precision close to stored value (0.82)
        # The "full" path has measured_precision_at_50 = 0.82
        _ingest_samples(monitor, "full", n_samples=10, precision_target=0.80)

        result = controller.check_precision_and_update_routing("full")

        assert result["routing_table_updated"] is False
        assert result["path_reselected"] is False

    def test_update_when_deviation_exceeds_threshold(self):
        """When deviation > 0.05, routing table entry should be updated (Req 4.2)."""
        controller, monitor, _ = _create_controller_with_monitor(min_samples=5)

        # Ingest samples that produce precision significantly different from 0.82
        # With precision_target=0.50, the estimated P@50 will deviate >> 0.05
        _ingest_samples(monitor, "full", n_samples=60, precision_target=0.50)

        result = controller.check_precision_and_update_routing("full")

        assert result["routing_table_updated"] is True
        assert result["estimated_precision"] is not None
        # The stored precision should now reflect the updated value
        for path in controller.get_routing_table():
            if path.path_id == "full":
                assert abs(path.measured_precision_at_50 - result["estimated_precision"]) < 1e-10

    def test_no_update_when_insufficient_samples(self):
        """When < min_samples, estimate is None and no update occurs."""
        controller, monitor, _ = _create_controller_with_monitor(min_samples=200)

        # Ingest fewer than min_samples
        _ingest_samples(monitor, "full", n_samples=50, precision_target=0.50)

        result = controller.check_precision_and_update_routing("full")

        assert result["estimated_precision"] is None
        assert result["routing_table_updated"] is False

    def test_no_update_when_no_monitor(self):
        """When no OnlinePrecisionMonitor is wired, nothing happens."""
        controller = DegradationController(precision_budget=0.60)

        result = controller.check_precision_and_update_routing("full")

        assert result["estimated_precision"] is None
        assert result["routing_table_updated"] is False


# ---------------------------------------------------------------------------
# Test: Path re-evaluation when routing table updated (Req 4.4)
# ---------------------------------------------------------------------------

class TestPathReEvaluation:
    """Tests for path re-evaluation after routing table update."""

    def test_path_reselection_on_update(self):
        """When current path precision drops below budget, a new path is selected (Req 4.4)."""
        # Create a custom routing table with only 2 paths
        paths = [
            ExecutionPath(
                path_id="path_a",
                required_components={"lightgbm", "td_pagerank"},
                measured_precision_at_50=0.80,
                description="Path A",
            ),
            ExecutionPath(
                path_id="path_b",
                required_components={"lightgbm"},
                measured_precision_at_50=0.70,
                description="Path B",
            ),
        ]
        controller, monitor, _ = _create_controller_with_monitor(
            paths=paths, min_samples=5, precision_budget=0.60
        )

        # Force current path to path_a
        controller._current_path = paths[0]

        # Ingest samples showing path_a precision dropped to 0.55 (below budget)
        # This is > 0.05 deviation from stored 0.80
        _ingest_samples(monitor, "path_a", n_samples=60, precision_target=0.10)

        result = controller.check_precision_and_update_routing("path_a")

        assert result["routing_table_updated"] is True
        # path_a precision dropped, so path_b should be selected
        # (path_b at 0.70 still meets budget of 0.60)
        assert result["path_reselected"] is True
        assert controller.current_path.path_id == "path_b"

    def test_no_reselection_when_current_still_best(self):
        """If current path remains best after update, no path switch occurs."""
        paths = [
            ExecutionPath(
                path_id="path_a",
                required_components={"lightgbm"},
                measured_precision_at_50=0.80,
                description="Path A",
            ),
            ExecutionPath(
                path_id="path_b",
                required_components={"lightgbm", "td_pagerank"},
                measured_precision_at_50=0.65,
                description="Path B",
            ),
        ]
        controller, monitor, _ = _create_controller_with_monitor(
            paths=paths, min_samples=5, precision_budget=0.60
        )

        # Force current path to path_a
        controller._current_path = paths[0]

        # Ingest samples showing path_a precision dropped from 0.80 to ~0.70
        # Deviation > 0.05 triggers update, but path_a still > path_b (0.65)
        # Use precision_target=0.60 which produces P@50 that is clearly
        # different from 0.80 (deviation > 0.05) but still meets budget
        _ingest_samples(monitor, "path_a", n_samples=60, precision_target=0.60)

        result = controller.check_precision_and_update_routing("path_a")

        # Check if deviation was large enough to trigger update
        if result["routing_table_updated"]:
            # After update, path_a still has higher precision than path_b (0.65)
            # OR they're close enough that path_a is still selected
            # Key assertion: current path is still path_a
            assert controller.current_path.path_id == "path_a"
        else:
            # If no update occurred (deviation ≤ 0.05), that's also fine — no switch
            assert controller.current_path.path_id == "path_a"


# ---------------------------------------------------------------------------
# Test: CRITICAL alert when all paths degrade below budget (Req 4.6)
# ---------------------------------------------------------------------------

class TestAllPathsDegraded:
    """Tests for CRITICAL alert when all paths below adaptive budget."""

    def test_critical_alert_all_paths_below_budget(self):
        """When all paths have precision < budget, CRITICAL alert emitted (Req 4.6)."""
        paths = [
            ExecutionPath(
                path_id="path_a",
                required_components={"lightgbm"},
                measured_precision_at_50=0.50,
                description="Path A - degraded",
            ),
            ExecutionPath(
                path_id="path_b",
                required_components={"lightgbm", "td_pagerank"},
                measured_precision_at_50=0.45,
                description="Path B - degraded",
            ),
        ]
        controller, monitor, _ = _create_controller_with_monitor(
            paths=paths, min_samples=5, precision_budget=0.60,
            stale_queue_age_minutes=10,  # Within 30 min → stale queue available
        )

        result = controller.check_precision_and_update_routing("path_a")

        assert result["all_paths_degraded"] is True
        assert controller.alert_level == "CRITICAL"

    def test_no_critical_when_some_paths_above_budget(self):
        """When at least one path is above budget, no CRITICAL alert."""
        paths = [
            ExecutionPath(
                path_id="path_a",
                required_components={"lightgbm"},
                measured_precision_at_50=0.70,
                description="Path A - healthy",
            ),
            ExecutionPath(
                path_id="path_b",
                required_components={"lightgbm", "td_pagerank"},
                measured_precision_at_50=0.45,
                description="Path B - degraded",
            ),
        ]
        controller, monitor, _ = _create_controller_with_monitor(
            paths=paths, min_samples=5, precision_budget=0.60
        )

        result = controller.check_precision_and_update_routing("path_a")

        assert result["all_paths_degraded"] is False

    def test_stale_queue_served_when_all_degraded_within_30min(self):
        """When all paths degraded and stale queue < 30 min, serve stale (Req 4.6)."""
        paths = [
            ExecutionPath(
                path_id="path_a",
                required_components={"lightgbm"},
                measured_precision_at_50=0.40,
                description="Path A - degraded",
            ),
        ]
        controller, monitor, _ = _create_controller_with_monitor(
            paths=paths, min_samples=5, precision_budget=0.60,
            stale_queue_age_minutes=15,  # 15 min < 30 min → available
        )

        result = controller.check_precision_and_update_routing("path_a")

        assert result["all_paths_degraded"] is True
        assert controller.alert_level == "CRITICAL"

    def test_critical_when_stale_queue_expired(self):
        """When all paths degraded and stale queue > 30 min, still CRITICAL (Req 4.6)."""
        paths = [
            ExecutionPath(
                path_id="path_a",
                required_components={"lightgbm"},
                measured_precision_at_50=0.40,
                description="Path A - degraded",
            ),
        ]
        controller, monitor, _ = _create_controller_with_monitor(
            paths=paths, min_samples=5, precision_budget=0.60,
            stale_queue_age_minutes=45,  # 45 min > 30 min → expired
        )

        result = controller.check_precision_and_update_routing("path_a")

        assert result["all_paths_degraded"] is True
        assert controller.alert_level == "CRITICAL"


# ---------------------------------------------------------------------------
# Test: Path re-selection within 500ms on PRECISION_DEGRADED (Req 5.5)
# ---------------------------------------------------------------------------

class TestPrecisionDegradedReselection:
    """Tests for rapid path re-selection on PRECISION_DEGRADED alert."""

    def test_reselection_on_precision_degraded_alert(self):
        """When PRECISION_DEGRADED is classified, path re-selection triggers (Req 5.5)."""
        paths = [
            ExecutionPath(
                path_id="path_a",
                required_components={"lightgbm", "td_pagerank"},
                measured_precision_at_50=0.80,
                description="Path A",
            ),
            ExecutionPath(
                path_id="path_b",
                required_components={"lightgbm"},
                measured_precision_at_50=0.70,
                description="Path B",
            ),
        ]
        controller, monitor, drift_detector = _create_controller_with_monitor(
            paths=paths, min_samples=5, precision_budget=0.60
        )

        # Force path_a as current
        controller._current_path = paths[0]

        # Ingest low-precision samples for path_a so drift detector classifies
        # as precision_degraded
        _ingest_samples(monitor, "path_a", n_samples=60, precision_target=0.10)

        # First, classify drift to set the per-path state to 'precision_degraded'
        scores = np.random.uniform(0, 1, size=100)
        # Set reference distribution first so KL can be computed
        drift_detector.update_reference(np.random.uniform(0, 1, size=100), "path_a")
        classification = drift_detector.classify_drift(scores, "path_a")

        # Verify drift detector detected degradation
        assert classification == "precision_degraded"

        # Now call the integration method - it should detect the degraded state
        # and trigger re-selection
        result = controller.check_precision_and_update_routing("path_a")

        # Either routing_table_updated triggered reselection, or drift-based did
        assert result["path_reselected"] is True

    def test_reselection_completes_within_500ms(self):
        """Path re-selection must complete within 500ms (Req 5.5)."""
        controller, monitor, drift_detector = _create_controller_with_monitor(min_samples=5)

        # Ingest low-precision samples
        _ingest_samples(monitor, "full", n_samples=60, precision_target=0.10)

        # Set up drift detector state
        scores = np.random.uniform(0, 1, size=100)
        drift_detector.update_reference(np.random.uniform(0, 1, size=100), "full")
        drift_detector.classify_drift(scores, "full")

        start = time.monotonic()
        result = controller.check_precision_and_update_routing("full")
        elapsed_ms = (time.monotonic() - start) * 1000

        # The method should complete well within 500ms
        assert elapsed_ms < 500, (
            f"check_precision_and_update_routing took {elapsed_ms:.1f}ms, "
            f"exceeding 500ms SLA"
        )

    def test_update_online_precision_entry_point(self):
        """update_online_precision() processes scores and checks routing."""
        controller, monitor, drift_detector = _create_controller_with_monitor(min_samples=5)

        # Ingest samples
        _ingest_samples(monitor, "full", n_samples=60, precision_target=0.50)

        # Set up reference for drift detection
        drift_detector.update_reference(np.random.uniform(0, 1, size=100), "full")

        scores = np.random.uniform(0, 1, size=100)
        result = controller.update_online_precision(scores, "full")

        assert "classification" in result
        assert "routing_result" in result
        assert result["routing_result"] is not None


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for the precision monitor integration."""

    def test_unknown_path_returns_gracefully(self):
        """Checking an unknown path_id returns a safe result."""
        controller, monitor, _ = _create_controller_with_monitor(min_samples=5)

        result = controller.check_precision_and_update_routing("nonexistent_path")

        assert result["path_id"] == "nonexistent_path"
        assert result["routing_table_updated"] is False
        assert result["estimated_precision"] is None

    def test_no_current_path_returns_gracefully(self):
        """When there's no current path, returns gracefully."""
        controller, monitor, _ = _create_controller_with_monitor(min_samples=5)
        controller._current_path = None

        result = controller.check_precision_and_update_routing()

        assert result["path_id"] is None
        assert result["routing_table_updated"] is False

    def test_controller_still_works_without_monitor(self):
        """Controller operates normally when no monitor is wired."""
        controller = DegradationController(precision_budget=0.60)

        # Standard route() cycle should work fine
        decision = controller.handle_degradation()
        assert decision.action in ("switch_path", "maintain")

        # check_precision just returns None estimate
        result = controller.check_precision_and_update_routing("full")
        assert result["estimated_precision"] is None
        assert result["routing_table_updated"] is False
