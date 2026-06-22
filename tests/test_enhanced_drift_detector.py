"""
Unit tests for EnhancedConceptDriftDetector.
Validates: Requirements 5.1, 5.2, 5.3, 5.4
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from models.degradation_controller import (
    EnhancedConceptDriftDetector,
    OnlinePrecisionMonitor,
    AdaptivePrecisionBudget,
)


def test_no_drift_when_no_reference_and_no_precision():
    """When no reference distribution exists and no precision data, classify as no_drift."""
    monitor = OnlinePrecisionMonitor(min_samples=5, window_size=50)
    budget = AdaptivePrecisionBudget(base_budget=0.60)
    detector = EnhancedConceptDriftDetector(
        kl_threshold=0.15,
        precision_monitor=monitor,
        precision_budget=budget,
    )
    scores = np.random.uniform(0, 1, 100)
    result = detector.classify_drift(scores, "path_A")
    assert result == "no_drift", f"Expected no_drift but got {result}"
    print("PASS: test_no_drift_when_no_reference_and_no_precision")


def test_per_path_isolation():
    """Drift state for one path does not affect another (Req 5.4)."""
    monitor = OnlinePrecisionMonitor(min_samples=5, window_size=50)
    budget = AdaptivePrecisionBudget(base_budget=0.60)
    detector = EnhancedConceptDriftDetector(
        kl_threshold=0.15,
        precision_monitor=monitor,
        precision_budget=budget,
    )

    # Classify drift on path_A
    scores_a = np.random.uniform(0, 1, 100)
    detector.classify_drift(scores_a, "path_A")

    # Classify drift on path_B
    scores_b = np.random.uniform(0, 1, 100)
    detector.classify_drift(scores_b, "path_B")

    # Verify separate state
    state_a = detector.per_path_state("path_A")
    state_b = detector.per_path_state("path_B")

    assert state_a["path_id"] == "path_A"
    assert state_b["path_id"] == "path_B"
    assert state_a["classification_count"] == 1
    assert state_b["classification_count"] == 1
    # Each path has its own detector instance
    assert "path_A" in detector.monitored_paths
    assert "path_B" in detector.monitored_paths
    print("PASS: test_per_path_isolation")


def test_benign_shift_when_kl_high_but_precision_above_budget():
    """When KL > 0.15 but precision above budget → benign_shift (Req 5.2)."""
    monitor = OnlinePrecisionMonitor(min_samples=5, window_size=50)
    budget = AdaptivePrecisionBudget(base_budget=0.60)
    detector = EnhancedConceptDriftDetector(
        kl_threshold=0.15,
        precision_monitor=monitor,
        precision_budget=budget,
    )

    # Set reference as low scores [0, 0.3]
    detector.update_reference(np.random.uniform(0, 0.3, 200), "path_C")

    # Feed enough samples with HIGH precision (all labels True for top scores)
    base_time = datetime(2024, 1, 1)
    for i in range(10):
        monitor.ingest_feedback(
            f"acc_{i}", 0.9 - i * 0.01, True,
            base_time + timedelta(seconds=i), "path_C"
        )

    # Send scores from a VERY different distribution to get high KL
    shifted_scores = np.random.uniform(0.7, 1.0, 100)
    result = detector.classify_drift(shifted_scores, "path_C")
    assert result == "benign_shift", f"Expected benign_shift but got {result}"
    print("PASS: test_benign_shift_when_kl_high_but_precision_above_budget")


def test_precision_degraded_when_precision_below_budget():
    """When precision < budget regardless of KL → precision_degraded (Req 5.3)."""
    monitor = OnlinePrecisionMonitor(min_samples=5, window_size=50)
    budget = AdaptivePrecisionBudget(base_budget=0.60)
    detector = EnhancedConceptDriftDetector(
        kl_threshold=0.15,
        precision_monitor=monitor,
        precision_budget=budget,
    )

    # Set reference distribution
    detector.update_reference(np.random.uniform(0, 1, 200), "path_D")

    # Feed samples where ALL labels are False (precision = 0, which is < 0.60 budget)
    base_time = datetime(2024, 1, 1)
    for i in range(10):
        monitor.ingest_feedback(
            f"acc_{i}", 0.9 - i * 0.05, False,
            base_time + timedelta(seconds=i), "path_D"
        )

    result = detector.classify_drift(np.random.uniform(0, 1, 100), "path_D")
    assert result == "precision_degraded", f"Expected precision_degraded but got {result}"
    print("PASS: test_precision_degraded_when_precision_below_budget")


def test_precision_degraded_even_with_low_kl():
    """Precision below budget triggers PRECISION_DEGRADED even when KL is low (Req 5.3)."""
    monitor = OnlinePrecisionMonitor(min_samples=5, window_size=50)
    budget = AdaptivePrecisionBudget(base_budget=0.60)
    detector = EnhancedConceptDriftDetector(
        kl_threshold=0.15,
        precision_monitor=monitor,
        precision_budget=budget,
    )

    # Set reference and use same distribution for scoring (KL will be ~0)
    reference_scores = np.random.uniform(0.3, 0.7, 200)
    detector.update_reference(reference_scores, "path_E")

    # Feed samples with zero precision (all False labels)
    base_time = datetime(2024, 1, 1)
    for i in range(10):
        monitor.ingest_feedback(
            f"acc_{i}", 0.9 - i * 0.05, False,
            base_time + timedelta(seconds=i), "path_E"
        )

    # Use similar distribution to keep KL low
    similar_scores = np.random.uniform(0.3, 0.7, 100)
    result = detector.classify_drift(similar_scores, "path_E")
    assert result == "precision_degraded", f"Expected precision_degraded but got {result}"
    print("PASS: test_precision_degraded_even_with_low_kl")


def test_per_path_state_returns_correct_info():
    """per_path_state returns correct metadata for a path."""
    monitor = OnlinePrecisionMonitor(min_samples=200, window_size=500)
    budget = AdaptivePrecisionBudget(base_budget=0.60)
    detector = EnhancedConceptDriftDetector(
        kl_threshold=0.15,
        precision_monitor=monitor,
        precision_budget=budget,
    )

    # Unmonitored path should get created with defaults
    state = detector.per_path_state("new_path")
    assert state["path_id"] == "new_path"
    assert state["has_reference"] is False
    assert state["last_kl"] == 0.0
    assert state["classification_count"] == 0
    assert state["last_classification"] is None

    # After classification
    detector.update_reference(np.random.uniform(0, 1, 200), "new_path")
    detector.classify_drift(np.random.uniform(0, 1, 100), "new_path")
    state = detector.per_path_state("new_path")
    assert state["has_reference"] is True
    assert state["classification_count"] == 1
    assert state["last_classification"] in ("no_drift", "benign_shift", "precision_degraded")
    print("PASS: test_per_path_state_returns_correct_info")


def test_cross_path_no_contamination():
    """Modifying one path's state should not change another's (Req 5.4)."""
    monitor = OnlinePrecisionMonitor(min_samples=5, window_size=50)
    budget = AdaptivePrecisionBudget(base_budget=0.60)
    detector = EnhancedConceptDriftDetector(
        kl_threshold=0.15,
        precision_monitor=monitor,
        precision_budget=budget,
    )

    # Set reference for path_X only
    detector.update_reference(np.random.uniform(0, 0.3, 200), "path_X")

    # Classify on path_X with shifted distribution (high KL)
    # Feed high precision data for path_X
    base_time = datetime(2024, 1, 1)
    for i in range(10):
        monitor.ingest_feedback(
            f"acc_{i}", 0.9 - i * 0.01, True,
            base_time + timedelta(seconds=i), "path_X"
        )
    result_x = detector.classify_drift(np.random.uniform(0.8, 1.0, 100), "path_X")

    # path_Y should have no reference, no history
    state_y = detector.per_path_state("path_Y")
    assert state_y["has_reference"] is False
    assert state_y["classification_count"] == 0

    # Classify on path_Y (no reference → KL=0 → no_drift)
    result_y = detector.classify_drift(np.random.uniform(0, 1, 100), "path_Y")
    assert result_y == "no_drift"

    # path_X state unaffected by path_Y classification
    state_x = detector.per_path_state("path_X")
    assert state_x["classification_count"] == 1
    assert state_x["has_reference"] is True
    print("PASS: test_cross_path_no_contamination")


def test_default_initialization():
    """EnhancedConceptDriftDetector works with default parameters."""
    detector = EnhancedConceptDriftDetector()
    assert detector.kl_threshold == 0.15
    assert detector.monitored_paths == []

    # Should work without explicit monitor/budget
    result = detector.classify_drift(np.random.uniform(0, 1, 50), "test_path")
    assert result in ("no_drift", "benign_shift", "precision_degraded")
    print("PASS: test_default_initialization")


if __name__ == "__main__":
    test_no_drift_when_no_reference_and_no_precision()
    test_per_path_isolation()
    test_benign_shift_when_kl_high_but_precision_above_budget()
    test_precision_degraded_when_precision_below_budget()
    test_precision_degraded_even_with_low_kl()
    test_per_path_state_returns_correct_info()
    test_cross_path_no_contamination()
    test_default_initialization()
    print("\n=== All EnhancedConceptDriftDetector tests passed! ===")
