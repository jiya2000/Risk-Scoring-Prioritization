"""
Unit tests for ResourceManager (models/resource_manager.py).

Tests all five methods of the ResourceManager:
- allocate_penalty_buffers: distinct memory buffers for symmetric vs learnable penalty
- route_topology_pipeline: GNN vs manual pipeline with different resource profiles
- release_degraded_resources: freeing memory from disabled components
- atomic_model_swap: atomic weight update without interrupting scoring
- utilization_report: resource utilization snapshot generation

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch
import threading

from models.resource_manager import ResourceManager, _PENALTY_BUFFER_SIZES, _TOPOLOGY_PIPELINE_PROFILES


class TestAllocatePenaltyBuffers:
    """Tests for allocate_penalty_buffers (Requirement 7.1)."""

    def test_symmetric_mode_returns_correct_structure(self):
        """Symmetric mode allocates a buffer with expected keys."""
        rm = ResourceManager()
        result = rm.allocate_penalty_buffers("symmetric")

        assert "buffer_id" in result
        assert "size_bytes" in result
        assert "mode" in result
        assert result["mode"] == "symmetric"
        assert result["size_bytes"] == _PENALTY_BUFFER_SIZES["symmetric"]

    def test_learnable_mode_returns_correct_structure(self):
        """Learnable mode allocates a buffer with expected keys."""
        rm = ResourceManager()
        result = rm.allocate_penalty_buffers("learnable")

        assert result["mode"] == "learnable"
        assert result["size_bytes"] == _PENALTY_BUFFER_SIZES["learnable"]

    def test_learnable_mode_allocates_more_memory_than_symmetric(self):
        """Learnable mode requires more memory than symmetric (distinct profiles)."""
        rm = ResourceManager()
        symmetric = rm.allocate_penalty_buffers("symmetric")
        learnable = rm.allocate_penalty_buffers("learnable")

        assert learnable["size_bytes"] > symmetric["size_bytes"]

    def test_invalid_mode_raises_value_error(self):
        """Invalid mode raises ValueError."""
        rm = ResourceManager()
        with pytest.raises(ValueError, match="Invalid penalty mode"):
            rm.allocate_penalty_buffers("unknown")

    def test_switching_modes_releases_previous_buffer(self):
        """Switching from one mode to another releases the previous buffer."""
        rm = ResourceManager()
        rm.allocate_penalty_buffers("symmetric")
        rm.allocate_penalty_buffers("learnable")

        # Only one penalty buffer should remain active
        penalty_buffers = [
            info for info in rm._allocated_buffers.values()
            if info.get("purpose") == "penalty_computation"
        ]
        assert len(penalty_buffers) == 1
        assert penalty_buffers[0]["mode"] == "learnable"

    def test_allocate_updates_active_components(self):
        """Allocation updates the set of active components."""
        rm = ResourceManager()
        rm.allocate_penalty_buffers("learnable")
        assert "learnable_scc_penalty" in rm.active_components
        assert "static_scc_penalty" not in rm.active_components

        rm.allocate_penalty_buffers("symmetric")
        assert "static_scc_penalty" in rm.active_components
        assert "learnable_scc_penalty" not in rm.active_components

    def test_allocate_records_snapshot(self):
        """Allocation records a ResourceUtilizationSnapshot."""
        rm = ResourceManager()
        rm.allocate_penalty_buffers("symmetric")

        assert len(rm.utilization_history) == 1
        snapshot = rm.utilization_history[0]
        assert snapshot.event_type == "path_switch"
        assert snapshot.memory_allocated_bytes > 0
        assert snapshot.active_path_id == "penalty_symmetric"


class TestRouteTopologyPipeline:
    """Tests for route_topology_pipeline (Requirement 7.2)."""

    def test_gnn_pipeline_returns_correct_profile(self):
        """GNN pipeline returns expected resource profile."""
        rm = ResourceManager()
        result = rm.route_topology_pipeline(use_gnn=True)

        assert result["pipeline_type"] == "gnn"
        assert result["estimated_memory_bytes"] == _TOPOLOGY_PIPELINE_PROFILES["gnn"]["estimated_memory_bytes"]
        assert result["estimated_latency_ms"] == _TOPOLOGY_PIPELINE_PROFILES["gnn"]["estimated_latency_ms"]
        assert result["processing_units"] == _TOPOLOGY_PIPELINE_PROFILES["gnn"]["processing_units"]

    def test_manual_pipeline_returns_correct_profile(self):
        """Manual pipeline returns expected resource profile."""
        rm = ResourceManager()
        result = rm.route_topology_pipeline(use_gnn=False)

        assert result["pipeline_type"] == "manual"
        assert result["estimated_memory_bytes"] == _TOPOLOGY_PIPELINE_PROFILES["manual"]["estimated_memory_bytes"]

    def test_gnn_uses_more_resources_than_manual(self):
        """GNN pipeline uses measurably more memory and processing units."""
        rm = ResourceManager()
        gnn_result = rm.route_topology_pipeline(use_gnn=True)

        rm2 = ResourceManager()
        manual_result = rm2.route_topology_pipeline(use_gnn=False)

        assert gnn_result["estimated_memory_bytes"] > manual_result["estimated_memory_bytes"]
        assert gnn_result["processing_units"] > manual_result["processing_units"]
        assert gnn_result["estimated_latency_ms"] > manual_result["estimated_latency_ms"]

    def test_switching_pipelines_computes_memory_delta(self):
        """Switching from manual to GNN shows positive memory delta."""
        rm = ResourceManager()
        rm.route_topology_pipeline(use_gnn=False)
        result = rm.route_topology_pipeline(use_gnn=True)

        expected_delta = (
            _TOPOLOGY_PIPELINE_PROFILES["gnn"]["estimated_memory_bytes"]
            - _TOPOLOGY_PIPELINE_PROFILES["manual"]["estimated_memory_bytes"]
        )
        assert result["memory_delta_bytes"] == expected_delta
        assert result["memory_delta_bytes"] > 0

    def test_switching_pipelines_updates_active_components(self):
        """Pipeline switch updates active component set."""
        rm = ResourceManager()
        rm.route_topology_pipeline(use_gnn=True)
        assert "gnn_topology" in rm.active_components

        rm.route_topology_pipeline(use_gnn=False)
        assert "gnn_topology" not in rm.active_components
        assert "manual_topology_vector" in rm.active_components

    def test_route_records_snapshot(self):
        """Pipeline routing records a utilization snapshot."""
        rm = ResourceManager()
        rm.route_topology_pipeline(use_gnn=True)

        assert len(rm.utilization_history) == 1
        snapshot = rm.utilization_history[0]
        assert snapshot.event_type == "path_switch"


class TestReleaseDegradedResources:
    """Tests for release_degraded_resources (Requirement 7.4)."""

    def test_releases_known_component_memory(self):
        """Known components are released with their profiled memory amounts."""
        rm = ResourceManager()
        result = rm.release_degraded_resources({"nlp_summarizer", "symbolic_rules"})

        assert "nlp_summarizer" in result
        assert "symbolic_rules" in result
        assert result["nlp_summarizer"] == 150 * 1024 * 1024
        assert result["symbolic_rules"] == 10 * 1024 * 1024

    def test_releases_unknown_component_with_nominal_amount(self):
        """Unknown components are released with a nominal 1MB amount."""
        rm = ResourceManager()
        result = rm.release_degraded_resources({"unknown_component"})

        assert "unknown_component" in result
        assert result["unknown_component"] == 1 * 1024 * 1024

    def test_removes_components_from_active_set(self):
        """Released components are removed from the active set."""
        rm = ResourceManager()
        rm._active_components.add("nlp_summarizer")
        rm._active_components.add("symbolic_rules")

        rm.release_degraded_resources({"nlp_summarizer"})

        assert "nlp_summarizer" not in rm.active_components
        assert "symbolic_rules" in rm.active_components

    def test_empty_set_releases_nothing(self):
        """Empty set of disabled components releases nothing."""
        rm = ResourceManager()
        result = rm.release_degraded_resources(set())

        assert result == {}

    def test_release_records_degradation_snapshot(self):
        """Resource release records a degradation-type snapshot."""
        rm = ResourceManager()
        rm.release_degraded_resources({"lightgbm"})

        assert len(rm.utilization_history) == 1
        snapshot = rm.utilization_history[0]
        assert snapshot.event_type == "degradation"
        assert snapshot.memory_released_bytes > 0
        assert snapshot.memory_allocated_bytes == 0

    def test_release_updates_total_memory_released(self):
        """Total memory released counter is updated."""
        rm = ResourceManager()
        rm.release_degraded_resources({"td_pagerank"})

        assert rm._total_memory_released == 30 * 1024 * 1024


class TestAtomicModelSwap:
    """Tests for atomic_model_swap (Requirement 7.6)."""

    def test_successful_swap_returns_true(self):
        """Successful swap with valid tensors returns True."""
        rm = ResourceManager()
        new_weights = {
            "layer1.weight": torch.randn(32, 5),
            "layer1.bias": torch.randn(32),
        }
        assert rm.atomic_model_swap(new_weights) is True

    def test_swap_updates_current_weights(self):
        """After swap, current_model_weights reflects the new weights."""
        rm = ResourceManager()
        new_weights = {"param": torch.tensor([1.0, 2.0, 3.0])}
        rm.atomic_model_swap(new_weights)

        assert rm.current_model_weights is not None
        assert "param" in rm.current_model_weights
        assert torch.allclose(rm.current_model_weights["param"], torch.tensor([1.0, 2.0, 3.0]))

    def test_swap_creates_independent_copy(self):
        """Swap creates a deep copy — modifying original doesn't affect stored weights."""
        rm = ResourceManager()
        original = torch.tensor([1.0, 2.0, 3.0])
        new_weights = {"param": original}
        rm.atomic_model_swap(new_weights)

        # Modify original tensor
        original[0] = 999.0

        # Stored weights should be unchanged
        assert rm.current_model_weights["param"][0].item() == 1.0

    def test_empty_weights_returns_false(self):
        """Empty weights dict returns False."""
        rm = ResourceManager()
        assert rm.atomic_model_swap({}) is False

    def test_non_tensor_value_returns_false(self):
        """Non-tensor values in weights dict returns False."""
        rm = ResourceManager()
        assert rm.atomic_model_swap({"param": [1, 2, 3]}) is False

    def test_swap_records_model_swap_snapshot(self):
        """Successful swap records a model_swap-type snapshot."""
        rm = ResourceManager()
        new_weights = {"w": torch.randn(10, 10)}
        rm.atomic_model_swap(new_weights)

        assert len(rm.utilization_history) == 1
        snapshot = rm.utilization_history[0]
        assert snapshot.event_type == "model_swap"
        assert snapshot.memory_allocated_bytes > 0

    def test_concurrent_swaps_are_thread_safe(self):
        """Multiple concurrent swaps don't corrupt state."""
        rm = ResourceManager()
        results = []

        def do_swap(value):
            weights = {"param": torch.tensor([float(value)])}
            success = rm.atomic_model_swap(weights)
            results.append(success)

        threads = [threading.Thread(target=do_swap, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All swaps should succeed (serialized by lock)
        assert all(results)
        # Final weight should be one of the values (last writer wins)
        assert rm.current_model_weights is not None
        assert "param" in rm.current_model_weights


class TestUtilizationReport:
    """Tests for utilization_report (Requirement 7.5)."""

    def test_empty_report_has_correct_structure(self):
        """Fresh manager produces a valid report structure."""
        rm = ResourceManager()
        report = rm.utilization_report()

        assert report["memory_allocated_bytes"] == 0
        assert report["memory_released_bytes"] == 0
        assert report["net_memory_bytes"] == 0
        assert report["active_components"] == []
        assert report["active_component_count"] == 0
        assert report["active_pipeline"] is None
        assert report["allocated_buffers"] == 0
        assert report["snapshot_count"] == 0
        assert report["snapshots"] == []
        assert "improvement_evidence" in report

    def test_report_reflects_operations(self):
        """Report reflects allocations, releases, and swaps."""
        rm = ResourceManager()
        rm.allocate_penalty_buffers("learnable")
        rm.route_topology_pipeline(use_gnn=True)
        rm.release_degraded_resources({"nlp_summarizer"})
        rm.atomic_model_swap({"w": torch.randn(5, 5)})

        report = rm.utilization_report()

        assert report["memory_allocated_bytes"] > 0
        assert report["memory_released_bytes"] > 0
        assert report["active_component_count"] > 0
        assert report["snapshot_count"] == 4
        assert report["active_pipeline"] == "gnn"

    def test_improvement_evidence_tracks_event_counts(self):
        """Improvement evidence correctly counts event types."""
        rm = ResourceManager()
        rm.allocate_penalty_buffers("symmetric")
        rm.route_topology_pipeline(use_gnn=False)
        rm.release_degraded_resources({"lightgbm"})
        rm.atomic_model_swap({"w": torch.randn(3)})

        report = rm.utilization_report()
        evidence = report["improvement_evidence"]

        assert evidence["path_switches"] == 2  # penalty + topology
        assert evidence["degradation_releases"] == 1
        assert evidence["model_swaps"] == 1
        assert evidence["total_memory_saved_bytes"] > 0
        assert evidence["demonstrates_reduced_memory_during_degradation"] is True
        assert evidence["demonstrates_atomic_swap_without_interruption"] is True

    def test_snapshots_contain_required_fields(self):
        """Each snapshot in the report contains all required fields per Req 7.3."""
        rm = ResourceManager()
        rm.allocate_penalty_buffers("learnable")

        report = rm.utilization_report()
        assert len(report["snapshots"]) == 1

        snapshot = report["snapshots"][0]
        assert "timestamp" in snapshot
        assert "active_path_id" in snapshot
        assert "memory_allocated_bytes" in snapshot
        assert "memory_released_bytes" in snapshot
        assert "computation_time_ms" in snapshot
        assert "active_component_count" in snapshot
        assert "event_type" in snapshot

    def test_report_shows_reduced_memory_during_degradation(self):
        """Report demonstrates concrete improvement: reduced memory during degradation."""
        rm = ResourceManager()
        # Start with full components
        rm.route_topology_pipeline(use_gnn=True)
        rm.allocate_penalty_buffers("learnable")

        # Degrade: release GNN and learnable components
        rm.release_degraded_resources({"gnn_topology", "learnable_scc_penalty"})

        report = rm.utilization_report()
        assert report["memory_released_bytes"] > 0
        evidence = report["improvement_evidence"]
        assert evidence["demonstrates_reduced_memory_during_degradation"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
