# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.

"""LLM-based orchestration controller.

:class:`LLMController` is the *planner* half of OmniFlow.  Given a user query
and the set of available training-free experts it emits an ordered list of
:class:`~omnimodal.core.types.ControlToken` objects — the wire protocol the
:class:`~omnimodal.orchestration.pipeline.OrchestrationPipeline` consumes.

Design properties
-----------------
* **LLM-first, heuristic fallback.**  When a real text LLM is bound
  (``config.use_real_llm`` True and an ``llm`` callable supplied) the controller
  asks it for a JSON plan and parses it.  On *any* failure — exception,
  timeout, malformed JSON, unknown expert name — it transparently degrades to a
  fully deterministic rule-based heuristic planner, so behaviour is always
  reproducible.
* **Expert dependency resolution.**  Regardless of what the LLM proposes, the
  hard partial order ``OCR -> LAYOUT_ANALYSIS -> CHART_READING`` (and the rule
  that terminal reasoning experts run last) is enforced by re-sequencing the
  emitted tokens.
* **Parallel vs. sequential.**  Tokens whose ``sequence_order`` is equal are
  declared parallel-eligible; strictly increasing ``sequence_order`` denotes a
  sequential dependency.  The pipeline groups by this field when executing.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Protocol, runtime_checkable

from omnimodal.core.types import (
    EXPERT_CAPABILITIES,
    EXPERT_DEPENDENCY_CHAIN,
    ControlToken,
    ExpertType,
    OmniConfig,
)

__all__ = ["LLMController", "PlannerContext", "LLMCallable"]


# ---------------------------------------------------------------------------
# LLM callable protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class LLMCallable(Protocol):
    """Async callable producing a completion from a system+user prompt."""

    async def __call__(self, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
# Planner context
# ---------------------------------------------------------------------------
class PlannerContext:
    """Lightweight, mutable context bag handed to the planner.

    Plain class (not a pydantic model) on purpose: it carries heterogeneous
    runtime payloads (file paths, prior outputs, etc.) that should not be
    serialised as part of the contract.  All fields default to empty so
    callers can construct it incrementally::

        ctx = PlannerContext()
        ctx.images = ["page1.png", "page2.png"]
        ctx.available_experts = [ExpertType.OCR, ExpertType.VQA]
    """

    __slots__ = (
        "images",
        "documents",
        "audio",
        "video",
        "available_experts",
        "history",
        "extra",
    )

    def __init__(
        self,
        images: list[str] | None = None,
        documents: list[str] | None = None,
        audio: list[str] | None = None,
        video: list[str] | None = None,
        available_experts: list[ExpertType] | None = None,
        history: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.images: list[str] = list(images) if images else []
        self.documents: list[str] = list(documents) if documents else []
        self.audio: list[str] = list(audio) if audio else []
        self.video: list[str] = list(video) if video else []
        self.available_experts: list[ExpertType] = (
            list(available_experts) if available_experts else []
        )
        self.history: list[dict[str, Any]] = list(history) if history else []
        self.extra: dict[str, Any] = dict(extra) if extra else {}

    def has_modality(self, kind: str) -> bool:
        """Return True when at least one asset of ``kind`` is present."""
        return bool(getattr(self, kind, None))

    def summary(self) -> dict[str, Any]:
        """Static, JSON-serialisable snapshot used to seed the LLM prompt."""
        return {
            "images": len(self.images),
            "documents": len(self.documents),
            "audio": len(self.audio),
            "video": len(self.video),
            "available_experts": [e.value for e in self.available_experts],
            "history_turns": len(self.history),
        }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class LLMController:
    """LLM-driven planner emitting :class:`ControlToken` sequences.

    Args:
        config: OmniFlow configuration.  ``config.use_real_llm`` and
            ``config.planning_timeout_ms`` are honoured here.
        available_experts: Experts the controller may schedule.  Defaults to
            every member of :class:`ExpertType`.
        llm: Optional async callable implementing :class:`LLMCallable`.  When
            ``config.use_real_llm`` is True *and* this is provided, the
            controller will consult the LLM; otherwise it uses the heuristic
            planner directly.
    """

    #: Reasoning experts that must run *after* all perception experts.
    _TERMINAL_EXPERTS: frozenset[ExpertType] = frozenset(
        {ExpertType.TEXT_REASON, ExpertType.VQA}
    )

    def __init__(
        self,
        config: OmniConfig,
        available_experts: list[ExpertType] | None = None,
        llm: LLMCallable | None = None,
    ) -> None:
        self.config = config
        self.available_experts: list[ExpertType] = (
            list(available_experts)
            if available_experts
            else list(ExpertType)
        )
        self.llm: LLMCallable | None = llm

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    def build_system_prompt(self) -> str:
        """Build the system prompt enumerating available experts + capabilities."""
        lines: list[str] = [
            "You are OmniFlow's orchestrator. Given a user query and a context",
            "summary, decide which training-free expert modules to invoke and",
            "in what order. Respond with ONLY a JSON array (no prose) of plans:",
            "",
            '  [{"expert": "<ExpertType value>", "params": {...}, "parallel_group": <int>}]',
            "",
            "Rules:",
            "- `expert` MUST be one of the listed ExpertType string values.",
            "- Experts with the same `parallel_group` run concurrently.",
            "- A higher `parallel_group` runs strictly AFTER lower groups.",
            "- Respect the hard dependency chain: OCR -> LayoutAnalysis ->",
            "  ChartReading (each must have a strictly greater parallel_group",
            "  than its predecessor).",
            "- Terminal reasoning experts (VQA, TextReason) run LAST.",
            "- Omit experts that are not needed for the query.",
            "",
            "Available experts:",
        ]
        for et in self.available_experts:
            cap = EXPERT_CAPABILITIES.get(et, "")
            lines.append(f"- {et.value}: {cap}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def plan(
        self,
        query: str,
        context: PlannerContext | None = None,
    ) -> list[ControlToken]:
        """Produce an ordered list of control tokens for ``query``.

        Falls back to the heuristic planner whenever the LLM path is disabled,
        unavailable, raises, or returns an unparseable response.
        """
        ctx = context or PlannerContext()
        ctx.available_experts = ctx.available_experts or list(self.available_experts)

        tokens: list[ControlToken] = []
        if self.config.use_real_llm and self.llm is not None:
            try:
                tokens = await asyncio.wait_for(
                    self._plan_with_llm(query, ctx),
                    timeout=self.config.planning_timeout_ms / 1000.0,
                )
            except Exception:
                # Deterministic degradation — never let a planner failure
                # propagate to the pipeline.
                tokens = []
        if not tokens:
            tokens = self._heuristic_plan(query, ctx)

        # Enforce the hard dependency partial order regardless of source.
        tokens = self._enforce_dependencies(tokens)
        # Compact the sequence_order so the lowest is 0 and gaps collapse.
        tokens = self._compact_order(tokens)
        return tokens

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------
    async def _plan_with_llm(
        self, query: str, ctx: PlannerContext
    ) -> list[ControlToken]:
        """Ask the bound LLM for a JSON plan and parse it to control tokens."""
        assert self.llm is not None  # noqa: S101 - narrow for type checkers
        system = self.build_system_prompt()
        user = (
            f"User query:\n{query}\n\n"
            f"Context summary:\n{json.dumps(ctx.summary())}\n\n"
            "Return the JSON plan now."
        )
        raw = await self.llm(system, user)
        return self._parse_llm_json(raw)

    def _parse_llm_json(self, raw: str) -> list[ControlToken]:
        """Best-effort extraction of a JSON plan array from LLM output."""
        if not raw or not raw.strip():
            return []

        # 1) Try direct parse.
        candidate: str | None = None
        try:
            data = json.loads(raw)
            candidate = raw
        except json.JSONDecodeError:
            candidate = None

        # 2) Fallback: extract the largest balanced JSON array substring.
        if candidate is None:
            match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
            if match:
                candidate = match.group(0)

        if candidate is None:
            return []

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []

        tokens: list[ControlToken] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            expert_name = str(item.get("expert", "")).strip()
            try:
                et = ExpertType(expert_name)
            except ValueError:
                # Unknown expert name from the LLM — skip.
                continue
            params = item.get("params", {}) or {}
            if not isinstance(params, dict):
                continue
            try:
                group = int(item.get("parallel_group", item.get("sequence_order", 0)))
            except (TypeError, ValueError):
                group = 0
            tokens.append(
                ControlToken(
                    expert_type=et,
                    params=params,
                    sequence_order=max(group, 0),
                )
            )
        return tokens

    # ------------------------------------------------------------------
    # Heuristic (rule-based) planner
    # ------------------------------------------------------------------
    def _heuristic_plan(
        self, query: str, ctx: PlannerContext
    ) -> list[ControlToken]:
        """Deterministic rule-based planner.

        Intent is inferred from keyword matching against the query and from
        which modalities are present in ``ctx``.  The resulting token list is
        later dependency-resolved + compacted by :meth:`plan`.
        """
        available = set(ctx.available_experts) or set(self.available_experts)
        q = (query or "").lower()

        has_images = ctx.has_modality("images")
        has_documents = ctx.has_modality("documents")
        has_audio = ctx.has_modality("audio")

        is_doc_intent = has_documents or any(
            k in q for k in (
                "document", "pdf", "page", "ocr", "read the", "scan",
                "invoice", "receipt", "form",
            )
        )
        is_chart_intent = any(
            k in q for k in (
                "chart", "graph", "plot", "bar", "pie", "axis", "trend",
            )
        )
        is_caption_intent = "caption" in q
        is_qa_intent = any(
            k in q for k in ("what", "describe", "explain", "question", "answer", "how")
        )
        is_audio_intent = has_audio or any(
            k in q for k in ("transcri", "speech", "audio", "voice", "recording")
        )

        stage0: list[ExpertType] = []
        if is_doc_intent and ExpertType.OCR in available:
            stage0.append(ExpertType.OCR)
        if has_images and ExpertType.IMAGE_CAPTION in available and (
            is_caption_intent or (is_qa_intent and not is_doc_intent)
        ):
            stage0.append(ExpertType.IMAGE_CAPTION)
        if is_audio_intent and ExpertType.ASR in available:
            stage0.append(ExpertType.ASR)
        if not stage0 and ExpertType.TEXT_REASON in available:
            # Pure-text question with no perception target.
            stage0.append(ExpertType.TEXT_REASON)

        tokens: list[ControlToken] = [
            ControlToken(expert_type=et, params={}, sequence_order=0)
            for et in stage0
        ]

        # Stage 1: structural analysis (depends on perception).
        stage1: list[ControlToken] = []
        if (
            is_doc_intent
            and ExpertType.LAYOUT_ANALYSIS in available
            and ExpertType.OCR in {t.expert_type for t in tokens}
        ):
            stage1.append(
                ControlToken(
                    expert_type=ExpertType.LAYOUT_ANALYSIS,
                    params={"depends_on": ExpertType.OCR.value},
                    sequence_order=1,
                )
            )
        if is_chart_intent and ExpertType.CHART_READING in available:
            dep = (
                ExpertType.LAYOUT_ANALYSIS.value
                if is_doc_intent
                else ExpertType.IMAGE_CAPTION.value
            )
            stage1.append(
                ControlToken(
                    expert_type=ExpertType.CHART_READING,
                    params={"depends_on": dep},
                    sequence_order=1,
                )
            )
        tokens.extend(stage1)

        # Stage 2: terminal reasoning/QA (depends on all upstream).
        upstream = {t.expert_type for t in tokens}
        if not upstream & self._TERMINAL_EXPERTS:
            terminal: ExpertType | None = None
            if has_images and ExpertType.VQA in available and is_qa_intent:
                terminal = ExpertType.VQA
            elif ExpertType.TEXT_REASON in available:
                terminal = ExpertType.TEXT_REASON
            if terminal is not None:
                tokens.append(
                    ControlToken(
                        expert_type=terminal,
                        params={"depends_on": "all"},
                        sequence_order=2,
                    )
                )
        return tokens

    # ------------------------------------------------------------------
    # Dependency resolution + ordering
    # ------------------------------------------------------------------
    def _enforce_dependencies(
        self, tokens: list[ControlToken]
    ) -> list[ControlToken]:
        """Re-sequence tokens to honour the hard expert dependency partial order.

        Guarantees:
        * Each member of :data:`EXPERT_DEPENDENCY_CHAIN` that is present has a
          strictly greater ``sequence_order`` than its chain predecessor.
        * Terminal reasoning experts run after every non-terminal expert.
        """
        if not tokens:
            return tokens

        # Index tokens by expert type. ``by_expert`` always holds the *current*
        # (possibly re-sequenced) view of each token.
        by_expert: dict[ExpertType, ControlToken] = {t.expert_type: t for t in tokens}

        # --- Chain ordering -------------------------------------------------
        # Walk the hard dependency chain in declaration order so the result is
        # deterministic regardless of frozenset/str-hash randomisation.
        present_chain = [et for et in EXPERT_DEPENDENCY_CHAIN if et in by_expert]
        # Start from the highest existing order to avoid collapsing too far.
        cur = max((t.sequence_order for t in tokens), default=0)
        prev_order = -1
        for et in present_chain:
            tok = by_expert[et]
            # Need strictly greater than the previous chain member's order.
            if tok.sequence_order <= prev_order:
                cur = max(cur, prev_order) + 1
                by_expert[et] = tok.model_copy(
                    update={"sequence_order": cur}, deep=False
                )
                prev_order = cur
            else:
                cur = max(cur, tok.sequence_order)
                prev_order = tok.sequence_order

        # --- Terminal experts run last -------------------------------------
        # Compute the max order among non-terminal tokens from the CURRENT
        # (post-chain) view in by_expert, then push terminal tokens past it.
        # Sort for deterministic processing (str-hash randomisation otherwise
        # makes frozenset iteration non-deterministic across processes).
        terminal_order = sorted(self._TERMINAL_EXPERTS, key=lambda e: e.value)
        max_non_terminal_order = max(
            (
                tok.sequence_order
                for et, tok in by_expert.items()
                if et not in self._TERMINAL_EXPERTS
            ),
            default=-1,
        )
        for et in terminal_order:
            if et in by_expert:
                tok = by_expert[et]
                if tok.sequence_order <= max_non_terminal_order:
                    new_order = max_non_terminal_order + 1
                    by_expert[et] = tok.model_copy(
                        update={"sequence_order": new_order}, deep=False
                    )
                    max_non_terminal_order = new_order

        # Preserve original input ordering in the returned list.
        return [by_expert[t.expert_type] for t in tokens]

    @staticmethod
    def _compact_order(tokens: list[ControlToken]) -> list[ControlToken]:
        """Renumber ``sequence_order`` to be contiguous starting at 0.

        Preserves the relative ordering (parallel groups stay grouped) but
        removes gaps so the pipeline sees stages 0, 1, 2, ...
        """
        if not tokens:
            return tokens
        unique_sorted = sorted({t.sequence_order for t in tokens})
        remap = {old: new for new, old in enumerate(unique_sorted)}
        return [
            t.model_copy(update={"sequence_order": remap[t.sequence_order]})
            if remap[t.sequence_order] != t.sequence_order
            else t
            for t in tokens
        ]
