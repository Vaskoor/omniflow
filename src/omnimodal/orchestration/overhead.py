# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.

"""Orchestration overhead analyser.

Training-free orchestration trades a single (expensive, monolithic) end-to-end
model for a graph of cheap, specialised experts coordinated by an LLM planner.
That coordination is not free: time is spent *generating control tokens*,
*dispatching* experts, and *shuttling* results through cross-modal memory.

:class:`OverheadAnalyzer` quantifies exactly that overhead from an
:class:`~omnimodal.core.types.OrchestrationTrace`, and compares the resulting
end-to-end latency against a hypothetical end-to-end trained baseline so a
caller can decide whether orchestration paid off for a given workload.
"""

from __future__ import annotations

from typing import Any

from omnimodal.core.types import OrchestrationTrace

__all__ = ["OverheadAnalyzer"]


# Small epsilon to keep divisions well-defined for near-zero latencies.
_EPS: float = 1e-9


class OverheadAnalyzer:
    """Measure orchestration overhead and compare against trained baselines.

    The analyser is *stateful*: :meth:`measure_overhead` caches its result so a
    subsequent :meth:`compare_with_baseline` call can contrast the measured
    orchestration latency against an externally-supplied end-to-end baseline
    without re-passing the trace.
    """

    def __init__(self) -> None:
        self._last_overhead: dict[str, float] = {}
        self._last_orchestration_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Overhead measurement
    # ------------------------------------------------------------------
    def measure_overhead(self, trace: OrchestrationTrace) -> dict[str, float]:
        """Decompose ``trace``'s latency into orchestration overhead components.

        Returns a dict with keys:

        * ``control_token_generation_ms`` — controller planning time.
        * ``expert_dispatch_ms`` — time spent scheduling + awaiting experts
          (excluding the experts' own work).
        * ``memory_io_ms`` — time spent writing to cross-modal memory.
        * ``total_overhead_ms`` — sum of the three above.
        * ``expert_work_ms`` — sum of the experts' self-reported latencies.
        * ``orchestration_latency_ms`` — the trace's full wall-clock latency.
        * ``overhead_ratio`` — ``total_overhead_ms / orchestration_latency_ms``.
        """
        planning_ms = float(trace.planning_ms or 0.0)
        dispatch_ms = 0.0
        memory_ms = 0.0
        expert_work_ms = 0.0

        for step in trace.steps:
            timing = step.timing_ms or {}
            dispatch_ms += float(timing.get("dispatch_ms", 0.0))
            memory_ms += float(timing.get("memory_ms", 0.0))
            expert_work_ms += float(timing.get("expert_work_ms", 0.0))

        total_overhead_ms = planning_ms + dispatch_ms + memory_ms
        orchestration_latency_ms = float(trace.total_latency_ms or 0.0)
        overhead_ratio = (
            total_overhead_ms / orchestration_latency_ms
            if orchestration_latency_ms > _EPS
            else 0.0
        )

        result: dict[str, float] = {
            "control_token_generation_ms": planning_ms,
            "expert_dispatch_ms": dispatch_ms,
            "memory_io_ms": memory_ms,
            "total_overhead_ms": total_overhead_ms,
            "expert_work_ms": expert_work_ms,
            "orchestration_latency_ms": orchestration_latency_ms,
            "overhead_ratio": overhead_ratio,
        }
        # Cache for subsequent baseline comparison.
        self._last_overhead = result
        self._last_orchestration_latency_ms = orchestration_latency_ms
        return result

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------
    def compare_with_baseline(
        self, end_to_end_latency_ms: float
    ) -> dict[str, Any]:
        """Compare the last-measured orchestration run vs. a trained baseline.

        Args:
            end_to_end_latency_ms: Wall-clock latency of the monolithic
                end-to-end trained model on the *same* workload.

        Returns a dict with keys:

        * ``orchestration_latency_ms`` — measured OmniFlow latency.
        * ``baseline_latency_ms`` — the supplied trained-baseline latency.
        * ``orchestration_overhead_ms`` — non-expert overhead inside OmniFlow.
        * ``net_benefit_ms`` — ``baseline - orchestration`` (positive = OmniFlow
          faster).
        * ``slowdown_ratio`` — ``orchestration / baseline`` (>1 means OmniFlow
          slower than the trained model).
        * ``overhead_ratio`` — orchestration overhead as a fraction of the
          OmniFlow run.
        * ``faster_than_baseline`` — convenience boolean.
        """
        baseline = float(end_to_end_latency_ms or 0.0)
        orchestration = self._last_orchestration_latency_ms
        overhead = float(self._last_overhead.get("total_overhead_ms", 0.0))
        overhead_ratio = float(self._last_overhead.get("overhead_ratio", 0.0))

        net_benefit_ms = baseline - orchestration
        slowdown_ratio = (
            orchestration / baseline if baseline > _EPS else float("inf")
        )
        faster = orchestration < baseline

        return {
            "orchestration_latency_ms": orchestration,
            "baseline_latency_ms": baseline,
            "orchestration_overhead_ms": overhead,
            "net_benefit_ms": net_benefit_ms,
            "slowdown_ratio": slowdown_ratio,
            "overhead_ratio": overhead_ratio,
            "faster_than_baseline": faster,
        }

    # ------------------------------------------------------------------
    # Convenience aggregations
    # ------------------------------------------------------------------
    @staticmethod
    def per_stage_breakdown(trace: OrchestrationTrace) -> list[dict[str, Any]]:
        """Return a per-stage timing breakdown for drill-down reporting.

        Each entry mirrors :meth:`measure_overhead` but scoped to a single
        :class:`~omnimodal.core.types.OrchestrationStep`, plus the step id and
        the expert types invoked.
        """
        breakdown: list[dict[str, Any]] = []
        for step in trace.steps:
            timing = step.timing_ms or {}
            dispatch = float(timing.get("dispatch_ms", 0.0))
            memory = float(timing.get("memory_ms", 0.0))
            work = float(timing.get("expert_work_ms", 0.0))
            total = dispatch + memory + work
            breakdown.append(
                {
                    "step_id": step.step_id,
                    "experts": [t.expert_type.value for t in step.control_tokens],
                    "dispatch_ms": dispatch,
                    "memory_io_ms": memory,
                    "expert_work_ms": work,
                    "step_total_ms": total,
                    "expert_calls": len(step.expert_responses),
                }
            )
        return breakdown
