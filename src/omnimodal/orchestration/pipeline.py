# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.

"""Orchestration execution pipeline.

:class:`OrchestrationPipeline` is the *executor* half of OmniFlow.  It takes a
:class:`~omnimodal.orchestration.controller.LLMController` plan (a list of
:class:`~omnimodal.core.types.ControlToken`) and:

1. Groups tokens by ``sequence_order`` into execution **stages**.
2. Runs every token in a stage **concurrently** (bounded by
   ``config.max_parallel_experts``), awaiting all before advancing.
3. Resolves each expert's input from the query, the raw assets, and the
   accumulated cross-modal memory (intermediate results flow forward).
4. Writes a :class:`~omnimodal.core.types.MemoryEntry` for every expert call.
5. Tracks per-stage latency + cost, returning a complete
   :class:`~omnimodal.core.types.OrchestrationTrace`.

Out of the box the pipeline ships deterministic **mock experts** so it runs
end-to-end with zero external services.  Real expert classes (shipped in
:mod:`omnimodal.experts`) drop in via an ``expert_registry`` mapping
``ExpertType -> ExpertProtocol``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol, runtime_checkable

from omnimodal.core.types import (
    EXPERT_PREFERRED_MODALITY,
    ControlToken,
    ExpertRequest,
    ExpertResponse,
    ExpertType,
    MemoryEntry,
    ModalityType,
    OmniConfig,
    OrchestrationStep,
    OrchestrationTrace,
)
from omnimodal.orchestration.controller import LLMController, PlannerContext

__all__ = [
    "OrchestrationPipeline",
    "ExpertProtocol",
    "MockExpert",
    "CrossModalMemory",
    "ExpertRegistry",
]


# ---------------------------------------------------------------------------
# Expert protocol + registry
# ---------------------------------------------------------------------------
@runtime_checkable
class ExpertProtocol(Protocol):
    """Async callable expert: ``ExpertRequest -> ExpertResponse``."""

    async def __call__(self, request: ExpertRequest) -> ExpertResponse: ...


#: Mapping from expert type to a callable expert instance.
ExpertRegistry = dict[ExpertType, ExpertProtocol]


# ---------------------------------------------------------------------------
# Cross-modal memory
# ---------------------------------------------------------------------------
class CrossModalMemory:
    """In-process, list-backed cross-modal working memory.

    A thin, well-typed store so the pipeline is self-contained.  When the
    dedicated :mod:`omnimodal.memory` subsystem ships, the pipeline will accept
    any duck-typed object exposing ``write(entry)`` and ``recent(n)``/``all()``.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: list[MemoryEntry] = []
        self._max_entries = max_entries

    def write(self, entry: MemoryEntry) -> None:
        """Append an entry, evicting the oldest when over capacity."""
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    def all(self) -> list[MemoryEntry]:
        """Return all entries in insertion order."""
        return list(self._entries)

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the ``n`` most recently written entries."""
        if n <= 0:
            return []
        return list(self._entries[-n:])

    def by_expert(self, expert_type: ExpertType) -> list[MemoryEntry]:
        """Return entries written by a given expert."""
        return [e for e in self._entries if e.source_expert == expert_type]

    def text_snapshot(self) -> str:
        """Render the whole memory as a single text blob for reasoning experts."""
        return "\n".join(
            f"[{e.source_expert.value} ({e.modality.value})] {e.text}"
            for e in self._entries
        )


# ---------------------------------------------------------------------------
# Deterministic mock expert
# ---------------------------------------------------------------------------
class MockExpert:
    """Deterministic fallback expert used when no real expert is registered.

    Produces a plausible textual projection of its input plus a fixed
    confidence so the full pipeline (controller -> mock experts -> memory ->
    trace) executes deterministically with no external services.
    """

    def __init__(
        self,
        expert_type: ExpertType,
        default_confidence: float = 0.5,
    ) -> None:
        self.expert_type = expert_type
        self.default_confidence = default_confidence

    async def __call__(self, request: ExpertRequest) -> ExpertResponse:
        start = time.perf_counter()
        # Simulate a tiny, deterministic amount of "work".
        await asyncio.sleep(0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        output = self._synthesize(request)
        return ExpertResponse(
            expert_type=self.expert_type,
            output=output,
            confidence=self.default_confidence,
            latency_ms=elapsed_ms,
        )

    def _synthesize(self, request: ExpertRequest) -> dict[str, Any]:
        """Produce a deterministic, structured mock output for this expert."""
        et = self.expert_type
        data = request.input_data
        textual = _to_text(data)
        if et is ExpertType.OCR:
            return {"text": f"[mock-ocr] {textual[:512]}", "boxes": []}
        if et is ExpertType.LAYOUT_ANALYSIS:
            return {
                "regions": [{"type": "paragraph", "text": textual[:256]}],
                "tree": {"root": "document", "children": []},
            }
        if et is ExpertType.CHART_READING:
            return {
                "chart_type": "bar",
                "series": [{"name": "mock", "points": []}],
                "summary": f"[mock-chart] {textual[:256]}",
            }
        if et is ExpertType.IMAGE_CAPTION:
            return {"caption": f"[mock-caption] an image related to: {textual[:256]}"}
        if et is ExpertType.VQA:
            return {
                "answer": f"[mock-vqa] based on the available evidence, "
                f"{textual[:256]}",
                "evidence_refs": [],
            }
        if et is ExpertType.TEXT_REASON:
            return {
                "answer": f"[mock-reason] synthesised from "
                f"{textual[:512]}",
                "rationale": "deterministic mock aggregation",
            }
        if et is ExpertType.ASR:
            return {"transcript": f"[mock-asr] {textual[:512]}", "language": "en"}
        return {"output": textual[:512]}


def _to_text(data: Any) -> str:
    """Best-effort textual projection of an arbitrary expert input payload."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, (list, tuple)):
        return " ".join(_to_text(x) for x in data)
    if isinstance(data, dict):
        # Pull the most informative field if present.
        for key in ("text", "answer", "caption", "transcript", "summary", "output"):
            if key in data and isinstance(data[key], str):
                return data[key]
        return " ".join(_to_text(v) for v in data.values())
    return str(data)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class OrchestrationPipeline:
    """Executes an :class:`LLMController` plan end-to-end.

    Args:
        config: OmniFlow configuration (cost model, parallelism, timeouts).
        controller: The planner producing control-token sequences.
        expert_registry: Optional mapping ``ExpertType -> ExpertProtocol``.
            Missing experts are transparently substituted by deterministic
            :class:`MockExpert` instances.
        memory: Optional cross-modal memory store.  Defaults to an internal
            :class:`CrossModalMemory` bounded by ``config.memory_max_entries``.
    """

    def __init__(
        self,
        config: OmniConfig,
        controller: LLMController,
        expert_registry: ExpertRegistry | None = None,
        memory: CrossModalMemory | Any = None,
    ) -> None:
        self.config = config
        self.controller = controller
        self.registry: ExpertRegistry = dict(expert_registry) if expert_registry else {}
        self.memory: CrossModalMemory = (
            memory
            if memory is not None
            else CrossModalMemory(max_entries=config.memory_max_entries)
        )
        # Internal cache of mock experts so we reuse instances per type.
        self._mocks: dict[ExpertType, MockExpert] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(
        self,
        query: str,
        images: list[str] | None = None,
        documents: list[str] | None = None,
    ) -> OrchestrationTrace:
        """Execute a full orchestration run.

        Args:
            query: The natural-language user query.
            images: Optional list of image asset references (paths/URLs).
            documents: Optional list of document references (paths/URLs).

        Returns:
            A complete :class:`OrchestrationTrace`.
        """
        run_start = time.perf_counter()

        ctx = PlannerContext(
            images=images,
            documents=documents,
            available_experts=list(self.controller.available_experts),
        )

        # --- Planning (control-token generation) --------------------------
        plan_start = time.perf_counter()
        plan_tokens = await self.controller.plan(query, ctx)
        planning_ms = (time.perf_counter() - plan_start) * 1000.0

        # --- Stage execution ----------------------------------------------
        stages = self._group_by_stage(plan_tokens)
        steps: list[OrchestrationStep] = []
        total_cost = 0.0
        expert_calls = 0

        # Seed memory with the user query so reasoning experts can see it.
        self.memory.write(
            MemoryEntry(
                text=f"User query: {query}",
                modality=ModalityType.TEXT,
                source_expert=ExpertType.TEXT_REASON,
            )
        )

        for stage_index, (order, tokens) in enumerate(stages):
            step, step_cost, step_calls = await self._execute_stage(
                stage_index, order, tokens, query, images or [], documents or []
            )
            steps.append(step)
            total_cost += step_cost
            expert_calls += step_calls

        total_latency_ms = (time.perf_counter() - run_start) * 1000.0

        return OrchestrationTrace(
            steps=steps,
            total_latency_ms=total_latency_ms,
            total_cost=total_cost,
            expert_calls=expert_calls,
            planning_ms=planning_ms,
            plan=plan_tokens,
        )

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------
    async def _execute_stage(
        self,
        stage_index: int,
        sequence_order: int,
        tokens: list[ControlToken],
        query: str,
        images: list[str],
        documents: list[str],
    ) -> tuple[OrchestrationStep, float, int]:
        """Run one (possibly parallel) stage of the plan."""
        stage_start = time.perf_counter()
        # Bound concurrency even within a single parallel stage.
        semaphore = asyncio.Semaphore(self.config.max_parallel_experts)

        async def _run_one(token: ControlToken) -> tuple[ControlToken, ExpertResponse]:
            async with semaphore:
                request = self._build_request(
                    token, query, images, documents
                )
                expert = self._resolve_expert(token.expert_type)
                try:
                    response = await asyncio.wait_for(
                        expert(request),
                        timeout=self.config.expert_timeout_ms / 1000.0,
                    )
                except Exception:
                    # Never let one expert's failure sink the stage.
                    response = await self._fallback_response(token.expert_type)
                return token, response

        dispatch_start = time.perf_counter()
        results = await asyncio.gather(*(_run_one(t) for t in tokens))
        dispatch_ms = (time.perf_counter() - dispatch_start) * 1000.0

        # --- Memory writes + cost ----------------------------------------
        memory_start = time.perf_counter()
        memory_writes: list[MemoryEntry] = []
        step_cost = 0.0
        for token, response in results:
            entry = self._to_memory_entry(token, response)
            self.memory.write(entry)
            memory_writes.append(entry)
            step_cost += self._cost_for(token.expert_type)
        memory_ms = (time.perf_counter() - memory_start) * 1000.0

        expert_work_ms = sum(r.latency_ms for _, r in results)

        step = OrchestrationStep(
            step_id=f"stage-{stage_index:02d}-order-{sequence_order}",
            control_tokens=[t for t, _ in results],
            expert_responses=[r for _, r in results],
            memory_writes=memory_writes,
            timing_ms={
                "expert_work_ms": expert_work_ms,
                "dispatch_ms": dispatch_ms,
                "memory_ms": memory_ms,
            },
        )
        return step, step_cost, len(results)

    # ------------------------------------------------------------------
    # Request building (intermediate-result routing)
    # ------------------------------------------------------------------
    def _build_request(
        self,
        token: ControlToken,
        query: str,
        images: list[str],
        documents: list[str],
    ) -> ExpertRequest:
        """Resolve the input payload for an expert from query + assets + memory.

        This is the *intermediate-result* routing layer: downstream experts
        receive upstream outputs via the cross-modal memory snapshot.
        """
        et = token.expert_type
        params = dict(token.params)
        snapshot = self.memory.text_snapshot()
        modality = EXPERT_PREFERRED_MODALITY.get(et, ModalityType.TEXT)

        if et is ExpertType.OCR:
            data: Any = documents[0] if documents else (images[0] if images else query)
        elif et is ExpertType.LAYOUT_ANALYSIS:
            ocr_entry = self.memory.by_expert(ExpertType.OCR)
            data = {
                "image": documents[0] if documents else (images[0] if images else None),
                "ocr_text": ocr_entry[-1].text if ocr_entry else "",
            }
        elif et is ExpertType.CHART_READING:
            layout = self.memory.by_expert(ExpertType.LAYOUT_ANALYSIS)
            caption = self.memory.by_expert(ExpertType.IMAGE_CAPTION)
            data = {
                "image": images[0] if images else (documents[0] if documents else None),
                "layout": layout[-1].text if layout else "",
                "caption": caption[-1].text if caption else "",
            }
        elif et is ExpertType.IMAGE_CAPTION:
            data = images[0] if images else query
        elif et is ExpertType.VQA:
            data = {
                "query": query,
                "image": images[0] if images else None,
                "context": snapshot,
            }
        elif et is ExpertType.TEXT_REASON:
            data = {"query": query, "context": snapshot}
        elif et is ExpertType.ASR:
            data = params.get("audio", query)
        else:
            data = {"query": query, "context": snapshot}

        return ExpertRequest(input_data=data, modality=modality, params=params)

    # ------------------------------------------------------------------
    # Expert resolution
    # ------------------------------------------------------------------
    def _resolve_expert(self, expert_type: ExpertType) -> ExpertProtocol:
        """Return the registered expert, else a deterministic mock."""
        expert = self.registry.get(expert_type)
        if expert is not None:
            return expert
        mock = self._mocks.get(expert_type)
        if mock is None:
            mock = MockExpert(
                expert_type=expert_type,
                default_confidence=self.config.default_confidence,
            )
            self._mocks[expert_type] = mock
        return mock

    async def _fallback_response(self, expert_type: ExpertType) -> ExpertResponse:
        """Produce a zero-cost failure response for a crashing expert."""
        return ExpertResponse(
            expert_type=expert_type,
            output={"error": "expert invocation failed"},
            confidence=0.0,
            latency_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Memory projection
    # ------------------------------------------------------------------
    @staticmethod
    def _to_memory_entry(
        token: ControlToken, response: ExpertResponse
    ) -> MemoryEntry:
        """Project an :class:`ExpertResponse` into a textual :class:`MemoryEntry`."""
        text = _to_text(response.output) or f"[{token.expert_type.value}] (empty)"
        return MemoryEntry(
            text=text,
            modality=EXPERT_PREFERRED_MODALITY.get(
                token.expert_type, ModalityType.TEXT
            ),
            source_expert=token.expert_type,
        )

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------
    def _cost_for(self, expert_type: ExpertType) -> float:
        """Look up the configured cost for one invocation of an expert."""
        return float(
            self.config.cost_per_expert_call.get(expert_type.value, 0.0)
        )

    # ------------------------------------------------------------------
    # Stage grouping
    # ------------------------------------------------------------------
    @staticmethod
    def _group_by_stage(
        tokens: list[ControlToken],
    ) -> list[tuple[int, list[ControlToken]]]:
        """Group tokens by ``sequence_order`` into ordered stages.

        Returns a list of ``(sequence_order, [tokens])`` tuples sorted by
        ascending order.  Tokens within a stage are eligible for parallel
        execution.
        """
        if not tokens:
            return []
        buckets: dict[int, list[ControlToken]] = {}
        for t in tokens:
            buckets.setdefault(t.sequence_order, []).append(t)
        return sorted(buckets.items())
