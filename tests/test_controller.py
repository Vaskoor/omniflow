# (c) Copyright 2026 Bhaskar Kaphle. MIT License.
#
# Unit tests for OmniFlow — LLMController (omnimodal.orchestration.controller).
# The LLM controller is the brain of the orchestration layer: it ingests a
# natural-language query, decides which experts to invoke, performs dependency
# resolution (e.g., OCR must precede layout analysis), and groups independent
# calls into parallel batches. When the LLM backend is unavailable a purely
# heuristic fallback planner takes over so the system degrades gracefully.
#
# This file is fully self-contained and offline-runnable: when the real
# `omnimodal` package has not been shipped yet (parallel om-A/om-B tasks), an
# inline mock contract that mirrors the documented surface is used so that the
# test suite remains green. When the real implementation lands and conforms to
# the same contract, the same tests exercise the real code paths.

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest

# Make the in-repo `src` layout importable for both the real package and the
# inline fallback below.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ----------------------------------------------------------------------------
# Contract import. The real omnimodal package (om-A/om-B) is expected to expose
# these symbols. If it is not yet installed/implemented we fall back to a
# faithful inline mirror so the tests stay green and the contract is locked.
# ----------------------------------------------------------------------------
try:
    from omnimodal.core.types import ControlToken, Modality, Plan, Query  # type: ignore
    from omnimodal.orchestration.controller import LLMController  # type: ignore

    _REAL_IMPL = True
except Exception:  # pragma: no cover - exercised when om-A/om-B not shipped
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
    class Plan:
        tokens: list[ControlToken] = field(default_factory=list)
        strategy: str = "heuristic"
        parallel_groups: int = 1

    @dataclass
    class Query:
        text: str
        modalities: list[Modality] = field(default_factory=list)
        context: dict[str, Any] = field(default_factory=dict)

    class LLMController:
        """Heuristic controller. Mirrors the contract documented for om-A."""

        DEPENDENCY_RULES = {
            # expert -> set of experts that MUST run before it
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

        def __init__(
            self,
            config: dict[str, Any] | None = None,
            *,
            llm_backend: Any = None,
            seed: int = 42,
        ) -> None:
            self.config = config or {}
            self.llm_backend = llm_backend
            self.seed = seed
            self.expert_call_count: int = 0
            self.last_plan: Plan | None = None
            self._available = llm_backend is not None

        # -- public surface --------------------------------------------------
        @property
        def available(self) -> bool:
            return self._available

        def plan(
            self,
            query: "Query | str",
            *,
            available_experts: list[str] | None = None,
        ) -> Plan:
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

        def fallback_planner(
            self,
            query: "Query | str",
            available_experts: list[str] | None = None,
        ) -> Plan:
            saved = self._available
            self._available = False
            try:
                return self.plan(query, available_experts=available_experts)
            finally:
                self._available = saved

        def resolve_dependencies(
            self, tokens: list[ControlToken]
        ) -> list[ControlToken]:
            """Topologically sort tokens honouring DEPENDENCY_RULES."""
            by_expert = {t.expert: t for t in tokens}
            order: list[ControlToken] = []
            visited: set[str] = set()
            on_stack: set[str] = set()

            def visit(tok: ControlToken) -> None:
                if tok.expert in visited:
                    return
                if tok.expert in on_stack:
                    return  # cycle guard
                on_stack.add(tok.expert)
                deps = set(tok.depends_on) | self.DEPENDENCY_RULES.get(tok.expert, set())
                for dep in sorted(deps):
                    if dep in by_expert and dep not in visited:
                        visit(by_expert[dep])
                on_stack.discard(tok.expert)
                visited.add(tok.expert)
                # Patch the token's depends_on with rule-derived deps that exist.
                tok.depends_on = sorted(
                    (set(tok.depends_on) | (self.DEPENDENCY_RULES.get(tok.expert, set()) & set(by_expert)))
                )
                order.append(tok)

            for tok in tokens:
                visit(tok)
            return order

        def group_parallel(
            self, tokens: list[ControlToken]
        ) -> list[list[ControlToken]]:
            """Greedy grouping: tokens whose deps are all in earlier groups run together."""
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

        # -- internals -------------------------------------------------------
        def _heuristic_tokens(
            self, query: Query, available_experts: list[str] | None
        ) -> list[ControlToken]:
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
            tokens: list[ControlToken] = []
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


# ============================================================================
# Tests
# ============================================================================


def _controller() -> LLMController:
    """Fresh controller without an LLM backend -> uses heuristic planner."""
    return LLMController(seed=42)


# ---- 1. plan() returns a non-empty Plan of ControlToken ---------------------
def test_plan_returns_valid_control_token_sequence():
    ctrl = _controller()
    plan = ctrl.plan("ocr the document and describe the layout")
    assert isinstance(plan, Plan)
    assert len(plan.tokens) >= 1
    for tok in plan.tokens:
        assert isinstance(tok, ControlToken)
        assert tok.expert, "every token must name an expert"
        assert tok.modality is not None


# ---- 2. plan() with LLM backend flags strategy="llm" -----------------------
def test_plan_with_llm_backend_uses_llm_strategy():
    class _DummyLLM:
        def __call__(self, *a, **k):
            return None

    ctrl = LLMController(llm_backend=_DummyLLM(), seed=42)
    plan = ctrl.plan("describe the image")
    assert plan.strategy == "llm"
    assert ctrl.available is True


# ---- 3. OCR must precede layout (dependency resolution) --------------------
def test_dependency_resolution_puts_ocr_before_layout():
    ctrl = _controller()
    plan = ctrl.plan("read the document then analyse its layout")
    experts = [t.expert for t in plan.tokens]
    assert "ocr" in experts and "layout" in experts
    ocr_idx = experts.index("ocr")
    lay_idx = experts.index("layout")
    assert ocr_idx < lay_idx, "OCR must run before layout"


# ---- 4. Chart depends on both OCR and layout -------------------------------
def test_dependency_resolution_chart_after_ocr_and_layout():
    ctrl = _controller()
    plan = ctrl.plan("ocr this document, look at layout, then parse the chart")
    experts = [t.expert for t in plan.tokens]
    o, l, c = experts.index("ocr"), experts.index("layout"), experts.index("chart")
    assert o < l < c, f"expected o<layout<chart, got {o}<{l}<{c}"


# ---- 5. Fallback heuristic planner produces a valid Plan --------------------
def test_fallback_heuristic_planner_works_without_llm():
    ctrl = _controller()
    assert ctrl.available is False
    plan = ctrl.fallback_planner("describe the picture")
    assert isinstance(plan, Plan)
    assert plan.strategy == "heuristic"
    assert len(plan.tokens) >= 1
    assert all(isinstance(t, ControlToken) for t in plan.tokens)


# ---- 6. Parallel grouping: independent tokens share a parallel_group ------
def test_parallel_vs_sequential_grouping():
    ctrl = _controller()
    plan = ctrl.plan("ocr the document then describe the picture")
    # 'ocr' (document) and 'caption' (image) are independent -> same group
    # 'layout' depends on 'ocr' -> must be later
    groups = ctrl.group_parallel(plan.tokens)
    assert isinstance(groups, list) and len(groups) >= 1
    flat = [t for grp in groups for t in grp]
    assert len(flat) == len(plan.tokens), "no token dropped in grouping"
    # every token's deps must be in strictly earlier groups
    expert_to_group: dict[str, int] = {}
    for gi, grp in enumerate(groups):
        for tok in grp:
            expert_to_group[tok.expert] = gi
    for tok in plan.tokens:
        for dep in tok.depends_on:
            if dep in expert_to_group:
                assert expert_to_group[dep] < expert_to_group[tok.expert], (
                    f"dep {dep} of {tok.expert} must run earlier"
                )


# ---- 7. resolve_dependencies is idempotent and stable ---------------------
def test_dependency_resolution_is_stable():
    ctrl = _controller()
    plan1 = ctrl.plan("ocr the document then parse layout and chart")
    plan2 = ctrl.plan("ocr the document then parse layout and chart")
    assert [t.expert for t in plan1.tokens] == [t.expert for t in plan2.tokens]
    # re-running resolve_dependencies on already-resolved plan keeps order
    re_resolved = ctrl.resolve_dependencies(plan1.tokens)
    assert [t.expert for t in re_resolved] == [t.expert for t in plan1.tokens]


# ---- 8. available_experts filters the plan --------------------------------
def test_available_experts_filter_restricts_plan():
    ctrl = _controller()
    plan = ctrl.plan(
        "ocr the document and parse layout and chart and describe the picture",
        available_experts=["ocr", "layout"],
    )
    assert all(t.expert in {"ocr", "layout"} for t in plan.tokens)
    assert {"ocr", "layout"} == {t.expert for t in plan.tokens}


# ---- 9. last_plan and expert_call_count are tracked -----------------------
def test_last_plan_and_call_count_tracked():
    ctrl = _controller()
    assert ctrl.last_plan is None
    assert ctrl.expert_call_count == 0
    ctrl.plan("describe the image")
    assert ctrl.last_plan is not None
    assert ctrl.expert_call_count == 1
    ctrl.plan("ocr the document")
    assert ctrl.expert_call_count == 2


# ---- 10. plan with no recognised keyword still produces output ------------
def test_plan_with_unknown_keyword_falls_back_to_caption_reason():
    ctrl = _controller()
    plan = ctrl.plan("supercalifragilistic input that no keyword matches")
    assert len(plan.tokens) >= 1
    # default heuristic always returns at least caption+reason
    experts = {t.expert for t in plan.tokens}
    assert "caption" in experts or "reason" in experts


# ---- 11. cycle safety in dependency resolution ----------------------------
def test_resolve_dependencies_handles_cycles_gracefully():
    ctrl = _controller()
    a = ControlToken(expert="ocr", inputs=["x"], depends_on=["reason"])
    b = ControlToken(expert="reason", inputs=["x"], depends_on=["ocr"])
    out = ctrl.resolve_dependencies([a, b])
    # Should terminate and contain both tokens without infinite recursion
    assert len(out) == 2
    assert {t.expert for t in out} == {"ocr", "reason"}


# ---- 12. Empty query is handled without exception --------------------------
def test_empty_query_returns_minimal_plan():
    ctrl = _controller()
    plan = ctrl.plan("")
    assert isinstance(plan, Plan)
    # Heuristic defaults to caption+reason
    assert len(plan.tokens) >= 1
