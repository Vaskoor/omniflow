# MIT License
#
# Copyright (c) 2024 OmniFlow contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""Expert base classes and registry for OmniFlow.

This module defines the :class:`Expert` abstract base class — the contract
every modal expert (OCR / layout / chart / caption / VQA / reasoning / ...)
must implement — together with the :class:`ExpertRegistry` that routes
:class:`ExpertRequest` objects to the correct expert instance.

Contract coupling: type imports come from ``omnimodal.core.types`` (shipped
by the parallel ``om-A`` task). If that package is not yet on the path the
module falls back to local minimal pydantic mirrors that share the exact
field names — so experts import cleanly both before and after ``om-A``
ships and produce identical observable behaviour in either mode.
"""
from __future__ import annotations

import abc
import hashlib
import json
import time
import uuid
from enum import Enum
from typing import Any, Iterable, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Contract types — try omnimodal.core.types first, fall back to local mirrors.
# The fallback mirrors the shipped contract field-for-field so that the
# package imports cleanly even when om-A is not on the path.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import path depends on om-A shipping status
    from omnimodal.core.types import (  # type: ignore[attr-defined]
        ControlToken as ControlToken,
        ExpertRequest as ExpertRequest,
        ExpertResponse as ExpertResponse,
        ExpertType as ExpertType,
        MemoryEntry as MemoryEntry,
        ModalityType as ModalityType,
    )
except ImportError:  # om-A not yet shipped — define local mirrors

    class ModalityType(str, Enum):
        TEXT = "text"
        IMAGE = "image"
        AUDIO = "audio"
        VIDEO = "video"
        LAYOUT = "layout"
        CHART = "chart"

    class ExpertType(str, Enum):
        OCR = "OCR"
        LAYOUT_ANALYSIS = "LayoutAnalysis"
        CHART_READING = "ChartReading"
        IMAGE_CAPTION = "ImageCaption"
        VQA = "VQA"
        TEXT_REASON = "TextReason"
        ASR = "ASR"

    def _coerce_expert_type(value: Any) -> "ExpertType":
        if isinstance(value, ExpertType):
            return value
        if isinstance(value, str):
            for et in ExpertType:
                if value == et.value or value.upper() == et.name:
                    return et
        raise ValueError(f"cannot coerce {value!r} to ExpertType")

    def _coerce_modality(value: Any) -> "ModalityType":
        if isinstance(value, ModalityType):
            return value
        if isinstance(value, str):
            for m in ModalityType:
                if value == m.value or value.upper() == m.name:
                    return m
        raise ValueError(f"cannot coerce {value!r} to ModalityType")

    class ControlToken(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        expert_type: ExpertType
        params: dict[str, Any] = Field(default_factory=dict)
        sequence_order: int = Field(default=0, ge=0)

    class ExpertRequest(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

        input_data: Any = None
        modality: ModalityType = ModalityType.TEXT
        params: dict[str, Any] = Field(default_factory=dict)

        @field_validator("modality", mode="before")
        @classmethod
        def _coerce_modality(cls, v: Any) -> Any:
            try:
                return _coerce_modality(v)
            except ValueError:
                return v

    class ExpertResponse(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

        expert_type: ExpertType
        output: Any = None
        confidence: float = Field(default=0.5, ge=0.0, le=1.0)
        latency_ms: float = Field(default=0.0, ge=0.0)

    class MemoryEntry(BaseModel):
        model_config = ConfigDict(extra="forbid")

        id: str = Field(default_factory=lambda: uuid.uuid4().hex)
        text: str
        modality: ModalityType
        source_expert: ExpertType
        timestamp: float = Field(default_factory=time.time)
        embedding_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Expert ABC
# ---------------------------------------------------------------------------
class Expert(abc.ABC):
    """Abstract base class for OmniFlow modal experts.

    Concrete experts override :pyattr:`expert_type`, :pyattr:`modalities` and
    :meth:`run`. The base class provides deterministic content hashing (used
    by mock backends) and latency timing utilities.
    """

    def __init__(self, *, backend: str = "mock", seed: int = 42, **kwargs: Any):
        self.backend = backend
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._options = kwargs

    # -- abstract surface ------------------------------------------------
    @property
    @abc.abstractmethod
    def expert_type(self) -> ExpertType:
        """ExpertType handled by this expert."""

    @property
    @abc.abstractmethod
    def modalities(self) -> list[ModalityType]:
        """Input modalities this expert accepts."""

    @abc.abstractmethod
    async def run(self, request: ExpertRequest) -> ExpertResponse:
        """Execute the expert on a request and return a response."""

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _content_hash(content: Any) -> str:
        """Deterministic SHA-256 hash of arbitrary content.

        Mock backends use this hash to seed numpy RandomState so that the
        same input always produces the same output.
        """
        if content is None:
            return "none" * 8
        if isinstance(content, (bytes, bytearray)):
            data = bytes(content)
        elif isinstance(content, str):
            data = content.encode("utf-8")
        elif isinstance(content, np.ndarray):
            data = content.tobytes()
        elif isinstance(content, (dict, list, tuple)):
            data = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
        else:
            data = repr(content).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def _rng_for(self, content: Any) -> np.random.RandomState:
        """Build a seeded RNG keyed off the content hash (deterministic)."""
        h = self._content_hash(content)
        try:
            seed_int = (int(h[:8], 16) ^ (self.seed * 0x9E3779B1)) & 0xFFFFFFFF
        except ValueError:
            seed_int = self.seed
        return np.random.RandomState(seed_int)

    @staticmethod
    def _measure_latency(t0: float) -> float:
        return (time.perf_counter() - t0) * 1000.0

    # -- response helpers ------------------------------------------------
    def _success_response(
        self,
        request: ExpertRequest,
        output: Any,
        confidence: float,
        latency_ms: float,
    ) -> ExpertResponse:
        return ExpertResponse(
            expert_type=self.expert_type,
            output=output,
            confidence=float(max(0.0, min(1.0, confidence))),
            latency_ms=float(max(0.0, latency_ms)),
        )

    def _error_response(
        self,
        request: ExpertRequest,
        error: str,
        latency_ms: float,
    ) -> ExpertResponse:
        """Build an error response — confidence 0, output carries the error."""
        return ExpertResponse(
            expert_type=self.expert_type,
            output={"error": error, "backend": self.backend},
            confidence=0.0,
            latency_ms=float(max(0.0, latency_ms)),
        )


# ---------------------------------------------------------------------------
# ExpertRegistry
# ---------------------------------------------------------------------------
class ExpertRegistry:
    """Routes :class:`ExpertRequest` objects to the right expert.

    Usage::

        registry = ExpertRegistry()
        registry.register(OCRExpert(backend="mock"))
        response = await registry.call(ExpertType.OCR, request)
    """

    def __init__(self, experts: Optional[Iterable[Expert]] = None):
        self._registry: dict[ExpertType, Expert] = {}
        if experts:
            for e in experts:
                self.register(e)

    # -- mutation --------------------------------------------------------
    def register(self, expert: Expert) -> None:
        if not isinstance(expert, Expert):
            raise TypeError(f"expected Expert instance, got {type(expert).__name__}")
        self._registry[expert.expert_type] = expert

    def unregister(self, expert_type: ExpertType | str) -> None:
        et = self._coerce(expert_type)
        self._registry.pop(et, None)

    # -- lookup ----------------------------------------------------------
    @staticmethod
    def _coerce(value: Any) -> ExpertType:
        if isinstance(value, ExpertType):
            return value
        if isinstance(value, str):
            for et in ExpertType:
                if value == et.value or value.upper() == et.name:
                    return et
        raise KeyError(f"unknown expert type: {value!r}")

    def get(self, expert_type: ExpertType | str) -> Expert:
        et = self._coerce(expert_type)
        if et not in self._registry:
            raise KeyError(f"no expert registered for {et.value}")
        return self._registry[et]

    def has(self, expert_type: ExpertType | str) -> bool:
        try:
            return self._coerce(expert_type) in self._registry
        except KeyError:
            return False

    def list(self) -> list[Expert]:
        return list(self._registry.values())

    def types(self) -> list[ExpertType]:
        return list(self._registry.keys())

    # -- execution -------------------------------------------------------
    async def call(
        self, expert_type: ExpertType | str, request: ExpertRequest
    ) -> ExpertResponse:
        """Invoke the expert registered for ``expert_type``."""
        expert = self.get(expert_type)
        return await expert.run(request)

    async def call_many(
        self, requests: list[tuple[ExpertType | str, ExpertRequest]]
    ) -> list[ExpertResponse]:
        """Fan-out — runs multiple expert requests concurrently."""
        import asyncio

        coros = [self.call(et, req) for et, req in requests]
        return await asyncio.gather(*coros)


__all__ = [
    "Expert",
    "ExpertRegistry",
    "ExpertType",
    "ExpertRequest",
    "ExpertResponse",
    "ModalityType",
    "MemoryEntry",
    "ControlToken",
]
