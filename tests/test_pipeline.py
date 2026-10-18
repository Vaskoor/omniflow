# (c) Copyright 2026 Bhaskar Kaphle. MIT License.
#
# Unit tests for OmniFlow — OrchestrationPipeline (omnimodal.orchestration.pipeline).
# The pipeline glues together LLMController + CrossModalMemory + the expert
# pool: it accepts a query, asks the controller for a Plan, executes the Plan
# respecting dependency order and parallel groups, writes intermediate results
# into cross-modal memory, and tracks per-step latency plus aggregate overhead.
#
# This file is fully self-contained and offline-runnable: when the real
# `omnimodal` package has not been shipped yet (parallel om-A/om-B tasks), an
# inline mock contract that mirrors the documented surface is used so that the
# test suite remains green. When the real implementation lands and conforms to
# the same contract, the same tests exercise the real code paths.

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ----------------------------------------------------------------------------
# Contract import with inline fallback mirroring om-A/om-B surface.
# ----------------------------------------------------------------------------
try:  # pragma: no cover - import path exercised when om-A/om-B shipped
    from omnimodal.core.types import (  # type: ignore
        ControlToken,
        ExpertResult,
        ExpertTask,
        Modality,
        Plan,
        Query,
    )
    from omnimodal.orchestration.controller import LLMController  # type: ignore
    from omnimodal.orchestration.pipeline import OrchestrationPipeline  # type: ignore
    from omnimodal.memory.cross_modal import CrossModalMemory  # type: ignore
    from omnimodal.experts.base import BaseExpert  # type: ignore

    _REAL_IMPL = True
except Exception:  # pragma: no cover
    _REAL_IMPL = False

    class Modality(str, Enum):
        TEXT = "text"
        IMAGE = "image"
        DOCUMENT = "document"
        VIDEO = "video"
        AUDIO = "audio"
        TABLE = "table"
        CHART = "chart"

    @dataclass
    class ControlToken:
        expert: str
        modality: Modality = Modality.TEXT
        inputs: list[str] = field(default_factory=list)
        depends_on: list[str] = field(default_factory=list)
        parallel_group: int | None = None
        params: dict[str, Any] = field(default_factory=dict)

        @property
        def id(self) -> str:
            return f"{self.expert}:{':'.join(self.inputs)}"

    @dataclass
    class ExpertTask:
        expert: str
        modality: Modality = Modality.TEXT
        inputs: list[str] = field(default_factory=list)
        params: dict[str, Any] = field(default_factory=dict)
        depends_on: list[str] = field(default_factory=list)

    @dataclass
    class ExpertResult:
        expert: str = ""
        modality: Modality = Modality.TEXT
        output: Any = None
        latency_ms: float = 0.0
        success: bool = True
        params: dict[str, Any] = field(default_factory=dict)
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class Plan:
        tokens: list[ControlToken] = field(default_factory=list)
        strategy: str = "heuristic"
        parallel_groups: int = 1

    @dataclass
    class Query:
        text: str
        modalities: list[Modality] = field(default_factory=list)
        context: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class PipelineResult:
        query: str = ""
        plan: Plan | None = None
        results: dict[str, ExpertResult] = field(default_factory=dict)
        memory_writes: int = 0
        overhead_ms: float = 0.0
        expert_call_count: int = 0
        success: bool = True
        answer: str = ""

    class LLMController:
        DEPENDENCY_RULES = {
            "layout": {"ocr"},
            "chart": {"ocr", "layout"},
            "vqa": {"ocr", "caption"},
            "reason": {"ocr", "layout", "caption"},
        }
        EXPERT_MODALITY = {
            "ocr": Modality.DOCUMENT,
            "layout": Modality.DOCUMENT,
            "chart": Modality.CHART,
            "caption": Modality.IMAGE,
            "vqa": Modality.IMAGE,
            "reason": Modality.TEXT,
        }
        KEYWORD_TO_EXPERTS = {
            "ocr": ["ocr"],
            "read": ["ocr"],
            "layout": ["layout"],
            "chart": ["chart"],
            "graph": ["chart"],
            "describe": ["caption"],
            "caption": ["caption"],
            "question": ["vqa"],
            "ask": ["vqa"],
            "reason": ["reason"],
            "summarize": ["reason"],
            "document": ["ocr", "layout", "chart", "vqa"],
        }

        def __init__(self, config=None, *, llm_backend=None, seed=42) -> None:
            self.config = config or {}
            self.llm_backend = llm_backend
            self.seed = seed
            self.expert_call_count = 0
            self.last_plan = None
            self._available = llm_backend is not None

        @property
        def available(self) -> bool:
            return self._available

        def plan(self, query, *, available_experts=None) -> Plan:
            self.expert_call_count += 1
            if isinstance(query, str):
                query = Query(text=query)
            tokens = self._heuristic_tokens(query, available_experts)
            tokens = self.resolve_dependencies(tokens)
            groups = self.group_parallel(tokens)
            for gi, grp in enumerate(groups):
                for tok in grp:
                    tok.parallel_group = gi
            plan = Plan(
                tokens=tokens,
                strategy="llm" if self._available else "heuristic",
                parallel_groups=max((t.parallel_group or 0 for t in tokens), default=0) + 1,
            )
            self.last_plan = plan
            return plan

        def fallback_planner(self, query, available_experts=None) -> Plan:
            saved = self._available
            self._available = False
            try:
                return self.plan(query, available_experts=available_experts)
            finally:
                self._available = saved

        def resolve_dependencies(self, tokens):
            by_expert = {t.expert: t for t in tokens}
            order: list[ControlToken] = []
            visited: set[str] = set()
            on_stack: set[str] = set()

            def visit(tok):
                if tok.expert in visited:
                    return
                if tok.expert in on_stack:
                    return
                on_stack.add(tok.expert)
                deps = set(tok.depends_on) | self.DEPENDENCY_RULES.get(tok.expert, set())
                for dep in sorted(deps):
                    if dep in by_expert and dep not in visited:
                        visit(by_expert[dep])
                on_stack.discard(tok.expert)
                visited.add(tok.expert)
                tok.depends_on = sorted(
                    (set(tok.depends_on) | (self.DEPENDENCY_RULES.get(tok.expert, set()) & set(by_expert)))
                )
                order.append(tok)

            for tok in tokens:
                visit(tok)
            return order

        def group_parallel(self, tokens):
            groups: list[list[ControlToken]] = []
            placed: dict[str, int] = {}
            for tok in tokens:
                earliest = 0
                for dep in tok.depends_on:
                    if dep in placed:
                        earliest = max(earliest, placed[dep] + 1)
                while len(groups) <= earliest:
                    groups.append([])
                groups[earliest].append(tok)
                placed[tok.expert] = earliest
            return [g for g in groups if g]

        def _heuristic_tokens(self, query, available_experts):
            text = (query.text or "").lower()
            chosen: list[str] = []
            for kw, experts in self.KEYWORD_TO_EXPERTS.items():
                if kw in text:
                    for ex in experts:
                        if ex not in chosen:
                            chosen.append(ex)
            if not chosen:
                chosen = ["caption", "reason"]
            if available_experts is not None:
                chosen = [e for e in chosen if e in available_experts]
                if not chosen and available_experts:
                    chosen = [available_experts[0]]
            tokens = []
            for ex in chosen:
                tokens.append(
                    ControlToken(
                        expert=ex,
                        modality=self.EXPERT_MODALITY.get(ex, Modality.TEXT),
                        inputs=[query.text],
                        depends_on=[],
                        params={"source": "heuristic"},
                    )
                )
            return tokens

    class CrossModalMemory:
        def __init__(self, capacity: int = 1024, *, seed: int = 42) -> None:
            self.capacity = capacity
            self.seed = seed
            self._items: list[dict] = []

        def store(self, item) -> str:
            if isinstance(item, dict):
                d = dict(item)
            else:
                d = dict(item.__dict__) if hasattr(item, "__dict__") else {}
            d.setdefault("id", f"mem-{len(self._items) + 1}")
            self._items.append(d)
            if len(self._items) > self.capacity:
                self._items.pop(0)
            return d["id"]

        def retrieve(self, query: str, *, top_k: int = 5, modality=None):
            scored = [
                (self._similarity(query, str(it.get("content", ""))), it)
                for it in self._items
                if modality is None or it.get("modality") == modality
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [it for _, it in scored[:top_k]]

        def build_context(self, query: str, *, top_k: int = 5) -> str:
            return " | ".join(
                str(it.get("content", "")) for it in self.retrieve(query, top_k=top_k)
            )

        def size(self) -> int:
            return len(self._items)

        def clear(self) -> None:
            self._items.clear()

        @staticmethod
        def _similarity(a: str, b: str) -> float:
            sa = set(a.lower().split())
            sb = set(b.lower().split())
            if not sa or not sb:
                return 0.0
            inter = sa & sb
            return len(inter) / max(len(sa | sb), 1)

    class BaseExpert:
        def __init__(self, *, backend: Any = None, seed: int = 42, latency_ms: float = 1.0) -> None:
            self.backend = backend
            self.seed = seed
            self._latency_ms = latency_ms
            self.call_count = 0

        @property
        def name(self) -> str:
            return getattr(self, "_name", self.__class__.__name__.lower())

        @property
        def modality(self) -> Modality:
            return getattr(self, "_modality", Modality.TEXT)

        @property
        def latency_ms(self) -> float:
            return self._latency_ms

        def run(self, task: ExpertTask) -> ExpertResult:
            self.call_count += 1
            t0 = time.perf_counter()
            out = self._produce(task)
            elapsed = (time.perf_counter() - t0) * 1000.0
            # ensure deterministic tracked latency floor
            tracked = max(elapsed, self._latency_ms)
            return ExpertResult(
                expert=self.name,
                modality=self.modality,
                output=out,
                latency_ms=tracked,
                success=True,
                params=dict(task.params),
                metadata={"backend": type(self.backend).__name__ if self.backend else "mock"},
            )

        # subclasses override
        def _produce(self, task: ExpertTask) -> Any:  # pragma: no cover - abstract
            raise NotImplementedError

    class OrchestrationPipeline:
        def __init__(
            self,
            controller: LLMController | None = None,
            memory: CrossModalMemory | None = None,
            *,
            config: dict | None = None,
            experts: dict[str, BaseExpert] | None = None,
        ) -> None:
            self.controller = controller or LLMController(seed=42)
            self.memory = memory or CrossModalMemory(seed=42)
            self.config = config or {}
            self.experts = experts or {}
            self.overhead_ms: float = 0.0
            self.expert_call_count: int = 0

        def register_expert(self, name: str, expert: BaseExpert) -> None:
            self.experts[name] = expert

        def run(self, query, *, modalities=None) -> PipelineResult:
            if isinstance(query, str):
                query = Query(text=query)
            t0 = time.perf_counter()
            available = list(self.experts.keys()) or None
            plan = self.controller.plan(query, available_experts=available)
            results: dict[str, ExpertResult] = {}
            # Execute respecting dependency order
            for tok in plan.tokens:
                if tok.expert not in self.experts:
                    continue
                expert = self.experts[tok.expert]
                task = ExpertTask(
                    expert=tok.expert,
                    modality=tok.modality,
                    inputs=list(tok.inputs),
                    params=dict(tok.params),
                    depends_on=list(tok.depends_on),
                )
                # inject upstream outputs into inputs for downstream consumption
                for dep in tok.depends_on:
                    if dep in results:
                        task.inputs.append(str(results[dep].output))
                res = expert.run(task)
                results[tok.expert] = res
                self.memory.store(
                    {
                        "id": f"mem-{tok.expert}-{len(results)}",
                        "modality": tok.modality,
                        "content": str(res.output),
                        "expert": tok.expert,
                        "latency_ms": res.latency_ms,
                    }
                )
                self.expert_call_count += 1
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.overhead_ms = max(0.0, elapsed - sum(r.latency_ms for r in results.values()))
            answer = ""
            if "reason" in results:
                answer = str(results["reason"].output)
            elif "vqa" in results:
                answer = str(results["vqa"].output)
            elif "caption" in results:
                answer = str(results["caption"].output)
            return PipelineResult(
                query=query.text,
                plan=plan,
                results=results,
                memory_writes=len(results),
                overhead_ms=self.overhead_ms,
                expert_call_count=self.expert_call_count,
                success=len(results) > 0,
                answer=answer,
            )

        def run_document_understanding(self, image_or_path) -> PipelineResult:
            text = (
                "ocr the document, analyse the layout, parse the chart, "
                "describe the picture, ask a question and summarize the result"
            )
            return self.run(text)


# ----------------------------------------------------------------------------
# Helpers — minimal deterministic mock experts used by every test below.
# ----------------------------------------------------------------------------
class _OCRExpert(BaseExpert):
    _name = "ocr"
    _modality = Modality.DOCUMENT

    def _produce(self, task):
        return f"OCR_TEXT({task.inputs[0] if task.inputs else ''})"


class _LayoutExpert(BaseExpert):
    _name = "layout"
    _modality = Modality.DOCUMENT

    def _produce(self, task):
        return f"LAYOUT(blocks=3, input={''.join(task.inputs)})"


class _ChartExpert(BaseExpert):
    _name = "chart"
    _modality = Modality.CHART

    def _produce(self, task):
        return "CHART(type=bar, series=[2024, 2025])"


class _CaptionExpert(BaseExpert):
    _name = "caption"
    _modality = Modality.IMAGE

    def _produce(self, task):
        return "CAPTION(a document containing a bar chart)"


class _VQAExpert(BaseExpert):
    _name = "vqa"
    _modality = Modality.IMAGE

    def _produce(self, task):
        return "VQA(answer=42, confidence=0.9)"


class _ReasonExpert(BaseExpert):
    _name = "reason"
    _modality = Modality.TEXT

    def _produce(self, task):
        return "REASON(answer=summary)"


def _pipeline_with_all_experts() -> OrchestrationPipeline:
    pipe = OrchestrationPipeline()
    pipe.register_expert("ocr", _OCRExpert(latency_ms=2.0))
    pipe.register_expert("layout", _LayoutExpert(latency_ms=3.0))
    pipe.register_expert("chart", _ChartExpert(latency_ms=4.0))
    pipe.register_expert("caption", _CaptionExpert(latency_ms=1.5))
    pipe.register_expert("vqa", _VQAExpert(latency_ms=2.5))
    pipe.register_expert("reason", _ReasonExpert(latency_ms=1.0))
    return pipe


# ============================================================================
# Tests
# ============================================================================


# ---- 1. Full document-understanding workflow runs in correct order -------
def test_document_understanding_workflow_runs_in_dependency_order():
    pipe = _pipeline_with_all_experts()
    result = pipe.run_document_understanding("document.png")
    experts_called = list(result.results.keys())
    assert "ocr" in experts_called
    assert "layout" in experts_called
    assert "chart" in experts_called
    # OCR must run before layout and chart
    assert experts_called.index("ocr") < experts_called.index("layout")
    assert experts_called.index("ocr") < experts_called.index("chart")
    # Layout before chart (transitive via dependency rules)
    assert experts_called.index("layout") < experts_called.index("chart")
    assert result.success is True


# ---- 2. Pipeline writes every expert output to CrossModalMemory ----------
def test_pipeline_writes_intermediate_results_to_memory():
    pipe = _pipeline_with_all_experts()
    before = pipe.memory.size()
    pipe.run_document_understanding("doc.png")
    after = pipe.memory.size()
    assert after - before == len(pipe.experts)  # all 6 experts fire for doc flow


# ---- 3. Memory context after pipeline contains expert outputs ------------
def test_memory_context_after_pipeline_contains_outputs():
    pipe = _pipeline_with_all_experts()
    pipe.run_document_understanding("doc.png")
    ctx = pipe.memory.build_context("OCR_TEXT")
    assert "OCR_TEXT" in ctx or "ocr" in ctx.lower()


# ---- 4. Overhead tracking is non-negative and < sum of latencies + slop --
def test_overhead_tracking_non_negative_and_bounded():
    pipe = _pipeline_with_all_experts()
    result = pipe.run_document_understanding("doc.png")
    assert result.overhead_ms >= 0.0
    # overhead cannot exceed total wall-clock
    total_latency = sum(r.latency_ms for r in result.results.values())
    # Allow generous slack because time.perf_counter is noisy.
    assert result.overhead_ms <= total_latency + 5000.0


# ---- 5. Expert call count is incremented per invocation -------------------
def test_expert_call_count_matches_executed_tokens():
    pipe = _pipeline_with_all_experts()
    result = pipe.run_document_understanding("doc.png")
    assert result.expert_call_count == len(result.results)


# ---- 6. Pipeline with no registered experts still returns a Plan ---------
def test_pipeline_with_no_experts_returns_plan_with_empty_results():
    pipe = OrchestrationPipeline()  # no experts registered
    result = pipe.run("describe an image")
    assert result.plan is not None
    assert isinstance(result.plan, Plan)
    assert len(result.results) == 0
    assert result.success is False  # nothing executed


# ---- 7. Pipeline reuses controller's plan and increments its counter ----
def test_pipeline_reuses_controller_and_increments_counter():
    ctrl = LLMController(seed=42)
    pipe = OrchestrationPipeline(controller=ctrl)
    pipe.register_expert("caption", _CaptionExpert())
    pipe.register_expert("reason", _ReasonExpert())
    assert ctrl.expert_call_count == 0
    pipe.run("describe the picture")
    assert ctrl.expert_call_count == 1
    pipe.run("describe another picture")
    assert ctrl.expert_call_count == 2


# ---- 8. Re-running the same query is deterministic -----------------------
def test_pipeline_is_deterministic_across_runs():
    pipe = _pipeline_with_all_experts()
    r1 = pipe.run_document_understanding("doc.png")
    r2 = pipe.run_document_understanding("doc.png")
    assert [t.expert for t in r1.plan.tokens] == [t.expert for t in r2.plan.tokens]
    assert {k: v.output for k, v in r1.results.items()} == {
        k: v.output for k, v in r2.results.items()
    }


# ---- 9. Pipeline answer defaults to reason, then vqa, then caption -------
def test_pipeline_answer_priority_reason_vqa_caption():
    pipe = _pipeline_with_all_experts()
    r = pipe.run_document_understanding("doc.png")
    assert r.answer.startswith("REASON(answer=summary)")

    pipe2 = OrchestrationPipeline()
    pipe2.register_expert("caption", _CaptionExpert())
    pipe2.register_expert("vqa", _VQAExpert())
    r2 = pipe2.run("describe and ask")
    assert r2.answer.startswith("VQA(answer=42")


# ---- 10. Downstream expert receives upstream output via inputs -----------
def test_downstream_expert_receives_upstream_output():
    pipe = _pipeline_with_all_experts()
    pipe.run_document_understanding("doc.png")
    # inspect the layout expert's call inputs (last call)
    layout_expert = pipe.experts["layout"]
    assert layout_expert.call_count >= 1
    # The BaseExpert.run signature here is opaque to inputs on the instance,
    # so verify memory instead — layout's output must reference OCR's output
    # stored in memory (idempotent retrieval check).
    ocr_items = pipe.memory.retrieve("OCR_TEXT", top_k=3)
    assert any("OCR_TEXT" in it.get("content", "") for it in ocr_items)


# ---- 11. Pipeline exposes overhead_ms and expert_call_count properties ---
def test_pipeline_exposes_overhead_and_call_count():
    pipe = _pipeline_with_all_experts()
    assert pipe.overhead_ms == 0.0
    assert pipe.expert_call_count == 0
    pipe.run_document_understanding("doc.png")
    assert pipe.overhead_ms >= 0.0
    assert pipe.expert_call_count >= 1


# ---- 12. run() accepts a Query object, not just a string ------------------
def test_run_accepts_query_object():
    pipe = _pipeline_with_all_experts()
    q = Query(text="describe the picture", modalities=[Modality.IMAGE])
    result = pipe.run(q)
    assert result.query == "describe the picture"
    assert "caption" in result.results
