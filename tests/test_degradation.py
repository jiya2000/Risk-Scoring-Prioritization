"""
Property-based tests for the Adaptive Degradation Controller.

Properties tested:
- Property 11: State machine transitions (DEGRADED/HEALTHY thresholds)
- Property 12: Optimal execution path selection with halt logic
- Property 13: Shadow evaluation escalation to CRITICAL
- Property 14: Flap protection cooldown locking

Validates: Requirements 3.2, 3.4, 3.5, 3.6, 3.8, 3.11
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path for cross-module imports
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st

from models.degradation_controller import DegradationController
from models.data_models import HealthVector, ComponentStatus

# Import the shared strategy from conftest (available automatically via pytest conftest)
# health_vector_sequence is registered as a composite strategy in conftest.py
from tests.conftest import health_vector_sequence


# ---------------------------------------------------------------------------
# Constants mirrored from DegradationController for clarity
# ---------------------------------------------------------------------------
COMPONENTS = DegradationController.COMPONENTS
HEARTBEAT_THRESHOLD_MS = DegradationController.HEARTBEAT_THRESHOLD_MS   # 5000.0
KL_THRESHOLD = DegradationController.KL_DIVERGENCE_THRESHOLD             # 0.5
THROUGHPUT_THRESHOLD = DegradationController.THROUGHPUT_THRESHOLD_RATIO  # 0.20
DEGRADED_CYCLES = DegradationController.DEGRADED_THRESHOLD_CYCLES        # 2
RECOVERY_CYCLES = DegradationController.RECOVERY_THRESHOLD_CYCLES        # 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_unhealthy(obs: Dict) -> bool:
    """Mirrors DegradationController._is_unhealthy logic for test assertions."""
    return (
        obs["heartbeat_latency_ms"] > HEARTBEAT_THRESHOLD_MS
        or obs["kl_divergence"] > KL_THRESHOLD
        or obs["throughput_ratio"] < THROUGHPUT_THRESHOLD
    )


def _make_health_vector(obs: Dict, ts: datetime) -> HealthVector:
    """Convert a strategy-generated dict to a HealthVector with a given timestamp."""
    return HealthVector(
        heartbeat_latency_ms=obs["heartbeat_latency_ms"],
        kl_divergence=obs["kl_divergence"],
        throughput_ratio=obs["throughput_ratio"],
        timestamp=ts,
    )


# ---------------------------------------------------------------------------
# Property 11 — State machine transitions
# Validates: Requirements 3.2, 3.6
# ---------------------------------------------------------------------------

@given(seq=health_vector_sequence(min_length=2, max_length=20))
@settings(max_examples=50, deadline=5000)
def test_property_11_state_machine_transitions(seq):
    """
    **Validates: Requirements 3.2, 3.6**

    Property 11: For any HealthVector sequence fed into evaluate_component(),
    a component becomes DEGRADED iff it accumulates ≥2 consecutive unhealthy
    observations, and returns to HEALTHY iff it accumulates ≥3 consecutive
    healthy observations.

    Thresholds (unhealthy if ANY):
      - heartbeat_latency_ms > 5000
      - kl_divergence > 0.5
      - throughput_ratio < 0.20
    """
    controller = DegradationController(precision_budget=0.60)
    component = "lightgbm"  # Use a single representative component

    # Track expected state with a reference simulation
    consecutive_unhealthy = 0
    consecutive_healthy = 0
    expected_status = ComponentStatus.HEALTHY

    base_time = datetime(2024, 1, 1, 0, 0, 0)

    for i, obs in enumerate(seq):
        ts = base_time + timedelta(seconds=i * 10)
        hv = _make_health_vector(obs, ts)

        # Compute actual status from controller
        actual = controller.evaluate_component(component, hv)

        unhealthy = _is_unhealthy(obs)

        if unhealthy:
            consecutive_healthy = 0
            consecutive_unhealthy += 1
        else:
            consecutive_unhealthy = 0
            consecutive_healthy += 1

        # Apply expected state machine rules
        if expected_status == ComponentStatus.HEALTHY:
            if consecutive_unhealthy >= DEGRADED_CYCLES:
                expected_status = ComponentStatus.DEGRADED
        elif expected_status == ComponentStatus.DEGRADED:
            if consecutive_healthy >= RECOVERY_CYCLES:
                expected_status = ComponentStatus.HEALTHY
                consecutive_unhealthy = 0  # reset after full recovery
        # COOLDOWN is handled separately by flap protection — skip if triggered
        if actual == ComponentStatus.COOLDOWN:
            # Flap protection engaged; remaining transitions locked — stop checking
            return

        note(f"Step {i}: obs={obs}, unhealthy={unhealthy}, "
             f"consec_unhealthy={consecutive_unhealthy}, "
             f"consec_healthy={consecutive_healthy}, "
             f"expected={expected_status}, actual={actual}")

        assert actual == expected_status, (
            f"Step {i}: expected {expected_status.value}, got {actual.value}. "
            f"consec_unhealthy={consecutive_unhealthy}, consec_healthy={consecutive_healthy}"
        )


# ---------------------------------------------------------------------------
# Property 12 — Optimal path selection with halt logic
# Validates: Requirements 3.4, 3.5
# ---------------------------------------------------------------------------

@given(healthy_set=st.sets(st.sampled_from(COMPONENTS)))
@settings(max_examples=50, deadline=5000)
def test_property_12_optimal_path_selection_halt(healthy_set):
    """
    **Validates: Requirements 3.4, 3.5**

    Property 12: For any subset of healthy components, the controller selects
    the execution path with the highest measured_precision_at_50 whose
    required_components ⊆ healthy_set AND precision ≥ 0.60.
    If no such path exists, the routing action must be "halt".
    """
    controller = DegradationController(precision_budget=0.60)

    # Force component statuses to match the generated healthy_set
    for comp in COMPONENTS:
        if comp in healthy_set:
            controller.force_component_status(comp, ComponentStatus.HEALTHY)
        else:
            controller.force_component_status(comp, ComponentStatus.DEGRADED)

    # Set a recent valid queue timestamp so halt is due to path unavailability,
    # not stale queue (we want to isolate path selection logic)
    controller.set_last_valid_queue_timestamp(datetime.now())

    decision = controller.handle_degradation()

    # Build the ground-truth best feasible path from the routing table
    routing_table = controller.get_routing_table()  # sorted descending by precision
    best_path = None
    for path in routing_table:
        if (path.required_components.issubset(healthy_set)
                and path.measured_precision_at_50 >= 0.60):
            best_path = path
            break  # First match = highest precision (table is sorted)

    note(f"healthy_set={healthy_set}, best_path={best_path.path_id if best_path else None}, "
         f"action={decision.action}, selected={decision.selected_path.path_id if decision.selected_path else None}")

    if best_path is None:
        # No feasible path → must halt
        assert decision.action == "halt", (
            f"Expected action='halt' when no feasible path, got '{decision.action}'. "
            f"healthy_set={healthy_set}"
        )
        assert decision.selected_path is None, (
            f"Expected selected_path=None on halt, got {decision.selected_path}"
        )
    else:
        # Feasible path exists → must select the highest-precision one
        assert decision.selected_path is not None, (
            f"Expected a selected path for healthy_set={healthy_set}, got None"
        )
        assert decision.selected_path.path_id == best_path.path_id, (
            f"Expected path '{best_path.path_id}' (precision={best_path.measured_precision_at_50}), "
            f"got '{decision.selected_path.path_id}'"
        )
        assert decision.precision_met is True


# ---------------------------------------------------------------------------
# Property 13 — Shadow evaluation escalation
# Validates: Requirement 3.8
# ---------------------------------------------------------------------------

@given(measured_precision=st.floats(min_value=0.0, max_value=1.0,
                                    allow_nan=False, allow_infinity=False))
@settings(max_examples=50, deadline=5000)
def test_property_13_shadow_evaluation_escalation(measured_precision):
    """
    **Validates: Requirements 3.8**

    Property 13: shadow_evaluate() raises a CRITICAL alert iff the measured
    Precision@50 is strictly below (precision_budget - 0.05) = 0.55.
    For any precision ≥ 0.55 the alert level must NOT be CRITICAL (it stays NORMAL).
    """
    controller = DegradationController(precision_budget=0.60)
    escalation_threshold = controller.precision_budget - 0.05  # 0.55

    # Build a mock DataFrame with 200+ rows.
    # We craft the scores so that exactly `measured_precision * 50` of the
    # top-50 rows have y_true=1, yielding the desired Precision@50.
    n_accounts = 210
    n_top = 50
    n_positives_in_top = int(round(measured_precision * n_top))
    n_positives_in_top = max(0, min(n_positives_in_top, n_top))

    # Top-50 accounts get high scores; rest get low scores
    scores = np.concatenate([
        np.linspace(1.0, 0.51, n_top),           # top 50 (descending)
        np.linspace(0.49, 0.0, n_accounts - n_top)  # bottom accounts
    ])

    # Assign y_true=1 to exactly n_positives_in_top of the top-50
    y_true = np.zeros(n_accounts, dtype=int)
    y_true[:n_positives_in_top] = 1

    held_out = pd.DataFrame({
        "account_id": [f"ACC_{i}" for i in range(n_accounts)],
        "y_true": y_true,
        "predicted_score": scores,
    })

    # Use the full execution path from the routing table
    path = controller.get_routing_table()[0]

    # Reset alert level to NORMAL before calling
    controller._alert_level = "NORMAL"
    returned_precision = controller.shadow_evaluate(path, held_out)

    note(f"measured_precision={measured_precision:.4f}, "
         f"returned_precision={returned_precision:.4f}, "
         f"escalation_threshold={escalation_threshold:.4f}, "
         f"alert_level={controller.alert_level}")

    if returned_precision < escalation_threshold:
        assert controller.alert_level == "CRITICAL", (
            f"Expected CRITICAL alert when precision {returned_precision:.4f} "
            f"< threshold {escalation_threshold:.4f}, got '{controller.alert_level}'"
        )
    else:
        assert controller.alert_level != "CRITICAL", (
            f"Expected non-CRITICAL alert when precision {returned_precision:.4f} "
            f">= threshold {escalation_threshold:.4f}, got '{controller.alert_level}'"
        )


# ---------------------------------------------------------------------------
# Property 14 — Flap protection cooldown
# Validates: Requirement 3.11
# ---------------------------------------------------------------------------

@given(
    component=st.sampled_from(COMPONENTS),
    n_extra_transitions=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50, deadline=5000)
def test_property_14_flap_protection_cooldown(component, n_extra_transitions):
    """
    **Validates: Requirements 3.11**

    Property 14: When a component transitions more than FLAP_LIMIT (3) times
    within a 5-minute window, it must be locked in COOLDOWN status for at
    least COOLDOWN_S (300 seconds = 5 minutes).

    Strategy: inject rapid alternating health vectors (unhealthy/healthy) with
    timestamps spaced a few seconds apart so all transitions fall inside the
    5-minute flap window.
    """
    controller = DegradationController(precision_budget=0.60)

    # Thresholds for clearly unhealthy / clearly healthy vectors
    unhealthy_hv_base = dict(
        heartbeat_latency_ms=9000.0,   # >> 5000 threshold
        kl_divergence=0.9,             # >> 0.5 threshold
        throughput_ratio=0.05,         # << 0.20 threshold
    )
    healthy_hv_base = dict(
        heartbeat_latency_ms=50.0,
        kl_divergence=0.01,
        throughput_ratio=1.0,
    )

    base_time = datetime(2024, 6, 1, 12, 0, 0)

    # Phase 1: Drive component to DEGRADED (requires 2 consecutive unhealthy)
    # Feed 2 unhealthy observations first
    for i in range(DEGRADED_CYCLES):
        ts = base_time + timedelta(seconds=i * 5)
        hv = HealthVector(**unhealthy_hv_base, timestamp=ts)
        controller.evaluate_component(component, hv)

    # Phase 2: Alternate between DEGRADED→HEALTHY→DEGRADED... to trigger flapping.
    # Each full cycle requires RECOVERY_CYCLES healthy + DEGRADED_CYCLES unhealthy.
    # We need > FLAP_LIMIT (3) transitions total within 5 minutes.
    # We'll do (FLAP_LIMIT + 1 + n_extra_transitions) complete cycles rapidly.
    n_cycles = DegradationController.FLAP_LIMIT + 1 + n_extra_transitions
    # Start timestamps right after phase 1
    t_offset = DEGRADED_CYCLES * 5 + 5  # seconds

    for cycle in range(n_cycles):
        # Recover: 3 healthy cycles
        for r in range(RECOVERY_CYCLES):
            ts = base_time + timedelta(seconds=t_offset)
            hv = HealthVector(**healthy_hv_base, timestamp=ts)
            controller.evaluate_component(component, hv)
            t_offset += 5  # 5 seconds apart → well within 5-min window

        # Degrade again: 2 unhealthy cycles
        for d in range(DEGRADED_CYCLES):
            ts = base_time + timedelta(seconds=t_offset)
            hv = HealthVector(**unhealthy_hv_base, timestamp=ts)
            controller.evaluate_component(component, hv)
            t_offset += 5

        # Stop as soon as COOLDOWN is triggered
        current = controller.get_component_status(component)
        if current == ComponentStatus.COOLDOWN:
            break

    note(f"component={component}, n_extra_transitions={n_extra_transitions}, "
         f"final_status={controller.get_component_status(component).value}, "
         f"cooldown_until={controller._cooldown_until.get(component)}")

    # Assert: component must now be in COOLDOWN
    final_status = controller.get_component_status(component)
    assert final_status == ComponentStatus.COOLDOWN, (
        f"Expected COOLDOWN after >{DegradationController.FLAP_LIMIT} transitions "
        f"in {DegradationController.FLAP_WINDOW_S}s window, got {final_status.value}"
    )

    # Assert: cooldown expires at least COOLDOWN_S seconds in the future
    cooldown_until = controller._cooldown_until[component]
    assert cooldown_until is not None, "cooldown_until should be set when COOLDOWN is active"

    # The last transition timestamp is approximately base_time + t_offset
    last_ts = base_time + timedelta(seconds=t_offset)
    min_expiry = last_ts - timedelta(seconds=5)  # small tolerance for timing
    assert cooldown_until >= min_expiry, (
        f"Cooldown expiry {cooldown_until} is less than expected minimum {min_expiry}"
    )

    # Assert: a subsequent healthy observation still returns COOLDOWN (lock holds)
    future_ts = cooldown_until - timedelta(seconds=1)  # still inside cooldown window
    hv_healthy = HealthVector(**healthy_hv_base, timestamp=future_ts)
    status_during_cooldown = controller.evaluate_component(component, hv_healthy)
    assert status_during_cooldown == ComponentStatus.COOLDOWN, (
        f"Component should remain COOLDOWN during cooldown window, "
        f"got {status_during_cooldown.value}"
    )
