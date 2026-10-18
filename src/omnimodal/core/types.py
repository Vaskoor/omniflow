# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# A copy of the License is included with this distribution.
#
# CONTRACT MODULE
# ---------------
# This file defines the **public data contract** for the entire OmniFlow
# framework.  Every sub-package (``core``, ``orchestration``, ``experts``,
# ``memory``, ``decoding``, ``streaming``, ``evaluation``) MUST speak these
# types verbatim.  In particular:
#
#   * ``ExpertType`` enum **values** are the names of the concrete expert
#     classes that ship in ``omnimodal.experts`` (e.g. ``ExpertType.OCR.value
#     == "OCR"`` <-> ``class OCR``).  This lets the pipeline dispatch experts by
#     name without a registry lookup.
#   * ``ControlToken`` is the *wire protocol* emitted by the LLM controller and
#     consumed by the orchestration pipeline.
#   * ``MemoryEntry`` is the canonical cross-modal memory record format.
#
# The contract is additive-only: new optional fields may be added, but the
# existing field names/types must never change shape.

"""Core type contract for OmniFlow.

All public data structures shared across the omni-modal orchestration stack.
Pydantic v2 is required; ``pydantic-settings`` is *optional* — when it is not
installed, ``OmniConfig`` degrades gracefully to a plain ``BaseModel`` with the
same field surface, so the package imports cleanly in minimal environments.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Optional dependency: pydantic-settings
# ---------------------------------------------------------------------------
# ``pydantic-settings`` ships ``BaseSettings``.  It is an *optional* dependency
# of OmniFlow (only needed when one wants environment-driven configuration).
# When it is missing we fall back to a shim that mirrors the same surface so
# ``OmniConfig`` can subclass ``BaseSettings`` unconditionally.
try:  # pragma: no cover - exercised only when pydantic-settings is present
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _SETTINGS_AVAILABLE: bool = True
except Exception:  # pragma: no cover - exercised when pydantic-settings absent
    _SETTINGS_AVAILABLE = False

    class BaseSettings(BaseModel):  # type: ignore[no-redef, misc]
        """Lightweight fallback for ``pydantic_settings.BaseSettings``.

        Behaves like a plain ``BaseModel``; environment binding is performed
        manually by :func:`omnimodal.core.config.load_config`.
        """

    def SettingsConfigDict(**kwargs: Any) -> dict[str, Any]:  # type: ignore[no-redef]
        """No-op stand-in for ``pydantic_settings.SettingsConfigDict``."""
        return dict(kwargs)


__all__ = [
    "ModalityType",
    "ExpertType",
    "ControlToken",
    "ExpertRequest",
    "ExpertResponse",
    "MemoryEntry",
    "OrchestrationStep",
    "OrchestrationTrace",
    "OmniConfig",
    "EXPERT_CAPABILITIES",
    "EXPERT_PREFERRED_MODALITY",
    "EXPERT_DEPENDENCY_CHAIN",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class ModalityType(str, Enum):
    """The discrete input/output modalities OmniFlow can route between."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    LAYOUT = "layout"
    CHART = "chart"


class ExpertType(str, Enum):
    """Catalogue of training-free expert modules.

    The string **value** of each member is the name of the concrete expert
    class living in :mod:`omnimodal.experts` (e.g. ``ExpertType.LAYOUT_ANALYSIS
    -> "LayoutAnalysis"``).  Keeping the value identical to the class name lets
    the pipeline resolve an expert by ``ExpertType`` without a lookup table.
    """

    OCR = "OCR"
    LAYOUT_ANALYSIS = "LayoutAnalysis"
    CHART_READING = "ChartReading"
    IMAGE_CAPTION = "ImageCaption"
    VQA = "VQA"
    TEXT_REASON = "TextReason"
    ASR = "ASR"


# ---------------------------------------------------------------------------
# Expert capability metadata
# ---------------------------------------------------------------------------
# Static capability descriptions consumed by the LLM controller when building
# its system prompt.  Kept here so the contract is the single source of truth.
EXPERT_CAPABILITIES: dict[ExpertType, str] = {
    ExpertType.OCR: (
        "Optical character recognition. Extracts raw text and reading order "
        "from document pages or scene images. Output is structured text with "
        "bounding boxes."
    ),
    ExpertType.LAYOUT_ANALYSIS: (
        "Document layout analysis. Segments a page into regions (paragraphs, "
        "headers, tables, figures) and emits a hierarchical layout tree. "
        "Depends on OCR output for text-region alignment."
    ),
    ExpertType.CHART_READING: (
        "Chart/figure interpretation. Parses rendered charts into a structured "
        "data table plus a natural-language summary. Depends on layout analysis "
        "for region localisation."
    ),
    ExpertType.IMAGE_CAPTION: (
        "Natural-language image captioning. Produces a concise descriptive "
        "caption for a single image. No upstream dependency."
    ),
    ExpertType.VQA: (
        "Visual question answering. Answers a free-form question about an image "
        "using the image plus any available captions/OCR text as context."
    ),
    ExpertType.TEXT_REASON: (
        "Pure-text reasoning / aggregation. Synthesises a final answer from "
        "the cross-modal memory. Terminal reasoning expert for most workflows."
    ),
    ExpertType.ASR: (
        "Automatic speech recognition. Transcribes an audio clip into text. "
        "No upstream dependency; downstream text can feed TEXT_REASON."
    ),
}

# The modality each expert primarily consumes (used for request routing).
EXPERT_PREFERRED_MODALITY: dict[ExpertType, ModalityType] = {
    ExpertType.OCR: ModalityType.IMAGE,
    ExpertType.LAYOUT_ANALYSIS: ModalityType.LAYOUT,
    ExpertType.CHART_READING: ModalityType.CHART,
    ExpertType.IMAGE_CAPTION: ModalityType.IMAGE,
    ExpertType.VQA: ModalityType.IMAGE,
    ExpertType.TEXT_REASON: ModalityType.TEXT,
    ExpertType.ASR: ModalityType.AUDIO,
}

# Hard dependency ordering — every planner MUST preserve this partial order.
EXPERT_DEPENDENCY_CHAIN: tuple[ExpertType, ...] = (
    ExpertType.OCR,
    ExpertType.LAYOUT_ANALYSIS,
    ExpertType.CHART_READING,
)


# ---------------------------------------------------------------------------
# Control protocol
# ---------------------------------------------------------------------------
class ControlToken(BaseModel):
    """A single instruction emitted by the controller, consumed by the pipeline.

    A list of :class:`ControlToken` objects is the *wire format* between the
    LLM orchestrator and the execution pipeline.  Tokens sharing the same
    ``sequence_order`` are eligible for **parallel** execution; tokens with a
    strictly greater ``sequence_order`` run **after** all tokens of a lower
    order complete (sequential dependency).

    Attributes:
        expert_type: Which expert to invoke.  ``ExpertType`` value matches the
            expert class name.
        params: Free-form expert parameters (e.g. ``{"language": "eng"}``,
            ``{"depends_on": "OCR"}``).  Opaque to the contract.
        sequence_order: Execution stage index.  Lower = earlier.  Equal values
            run concurrently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    expert_type: ExpertType
    params: dict[str, Any] = Field(default_factory=dict)
    sequence_order: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Expert I/O
# ---------------------------------------------------------------------------
class ExpertRequest(BaseModel):
    """Request envelope handed to an expert by the pipeline.

    Attributes:
        input_data: The raw payload for this expert call.  May be a file path,
            a list of paths, prior-stage structured output, or plain text.
        modality: Primary modality of ``input_data``.
        params: Expert-specific parameters merged from the control token and
            the pipeline's working context.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_data: Any = None
    modality: ModalityType = ModalityType.TEXT
    params: dict[str, Any] = Field(default_factory=dict)


class ExpertResponse(BaseModel):
    """Structured result returned by an expert.

    Attributes:
        expert_type: Expert that produced this response.
        output: Expert-specific structured payload (text, layout tree, chart
            table, answer string, ...).
        confidence: Calibrated confidence in ``[0.0, 1.0]``.
        latency_ms: Wall-clock latency of the expert call in milliseconds.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    expert_type: ExpertType
    output: Any = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# Cross-modal memory
# ---------------------------------------------------------------------------
class MemoryEntry(BaseModel):
    """A single record in the cross-modal working memory.

    Memory entries are written by the pipeline after every expert call and
    read by downstream experts (e.g. ``TEXT_REASON`` aggregates over all
    entries to synthesise a final answer).

    Attributes:
        id: Stable unique identifier (UUID4 hex by default).
        text: Natural-language rendering of the stored fact.  Experts that
            emit structured output are expected to also provide a textual
            projection so the text-reasoning expert can consume them.
        modality: Modality the fact originated from.
        source_expert: Expert that wrote the entry.
        timestamp: POSIX timestamp (seconds, UTC) of the write.
        embedding_id: Optional identifier of a dense embedding stored
            out-of-band by the memory subsystem.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    modality: ModalityType
    source_expert: ExpertType
    timestamp: float = Field(
        default_factory=lambda: datetime.now(UTC).timestamp()
    )
    embedding_id: str | None = None


# ---------------------------------------------------------------------------
# Orchestration trace
# ---------------------------------------------------------------------------
class OrchestrationStep(BaseModel):
    """One execution stage of an orchestration run.

    Attributes:
        step_id: Stable step identifier.
        control_tokens: Tokens scheduled at this step.
        expert_responses: Responses returned by the experts invoked here.
        memory_writes: Memory entries written as a side effect.
        timing_ms: Component timing breakdown in milliseconds.  Recognised
            keys: ``"expert_work_ms"``, ``"dispatch_ms"``, ``"memory_ms"``.
    """

    model_config = ConfigDict(extra="forbid")

    step_id: str
    control_tokens: list[ControlToken] = Field(default_factory=list)
    expert_responses: list[ExpertResponse] = Field(default_factory=list)
    memory_writes: list[MemoryEntry] = Field(default_factory=list)
    timing_ms: dict[str, float] = Field(default_factory=dict)


class OrchestrationTrace(BaseModel):
    """End-to-end trace of a single :meth:`OrchestrationPipeline.run` call.

    Attributes:
        steps: Ordered list of execution stages.
        total_latency_ms: Wall-clock latency of the whole run, including
            planning overhead.
        total_cost: Estimated monetary cost (currency-agnostic float).
        expert_calls: Total number of expert invocations across all steps.
        planning_ms: Time spent generating control tokens (controller overhead).
        plan: The control-token plan that produced this trace (for replay).
    """

    model_config = ConfigDict(extra="forbid")

    steps: list[OrchestrationStep] = Field(default_factory=list)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    total_cost: float = Field(default=0.0, ge=0.0)
    expert_calls: int = Field(default=0, ge=0)
    planning_ms: float = Field(default=0.0, ge=0.0)
    plan: list[ControlToken] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _default_expert_costs() -> dict[str, float]:
    """Per-expert default cost (currency-agnostic) per single invocation."""
    return {et.value: 0.0 for et in ExpertType}


class OmniConfig(BaseSettings):
    """Top-level OmniFlow configuration.

    Loadable from a YAML file via :func:`omnimodal.core.config.load_config`.
    All fields have deterministic defaults so the framework runs out-of-the-box
    with no external services.
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIFLOW_",
        case_sensitive=False,
        extra="ignore",
    )  # type: ignore[arg-type]

    # Framework identity.
    version: str = "0.1.0"

    # LLM orchestrator settings.  When ``use_real_llm`` is False, or when the
    # bound LLM callable raises / returns unparseable output, the controller
    # transparently falls back to the deterministic heuristic planner.
    use_real_llm: bool = False
    llm_model: str = "glm-4.6"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    planning_timeout_ms: float = 5000.0

    # Execution limits.
    max_parallel_experts: int = Field(default=4, ge=1)
    expert_timeout_ms: float = 30_000.0

    # Cost model — per-expert cost per single call (currency-agnostic).
    cost_per_expert_call: dict[str, float] = Field(
        default_factory=_default_expert_costs
    )

    # Determinism + observability.
    default_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_overhead_tracking: bool = True
    memory_max_entries: int = Field(default=1000, ge=1)
    random_seed: int = 0

    # Expert availability flags (the concrete expert classes in
    # ``omnimodal.experts`` may toggle these once they ship).
    available_experts: list[str] = Field(
        default_factory=lambda: [et.value for et in ExpertType]
    )
