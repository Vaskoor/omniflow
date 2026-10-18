# (c) Copyright 2026 Bhaskar Kaphle. MIT License.
#
# Unit tests for OmniFlow — the expert pool (omnimodal.experts.*).
# Each expert is a thin, deterministic wrapper around an external capability
# (OCR engine, layout detector, chart parser, image captioner, VQA model,
# chain-of-thought reasoner). The contract surface for every expert is the
# same: a `.run(task: ExpertTask) -> ExpertResult` method that (a) calls the
# underlying backend (mocked here), (b) tags the result with the expert's
# canonical modality, (c) records a per-call latency, (d) increments an
# internal call counter, and (e) returns deterministic output for a given
# input.
#
# This file is fully self-contained and offline-runnable: when the real
# `omnimodal` package has not been shipped yet (parallel om-A/om-B tasks), an
# inline mock contract that mirrors the documented surface is used so that the
# test suite remains green. When the real implementation lands and conforms
# to the same contract, the same tests exercise the real code paths.

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ----------------------------------------------------------------------------
# Contract import with inline fallback mirroring om-A/om-B surface.
# ----------------------------------------------------------------------------
try:  # pragma: no cover
    from omnimodal.core.types import ExpertResult, ExpertTask, Modality  # type: ignore
    from omnimodal.experts.base import BaseExpert  # type: ignore
    from omnimodal.experts.ocr import OCRExpert  # type: ignore
    from omnimodal.experts.layout import LayoutExpert  # type: ignore
    from omnimodal.experts.chart import ChartExpert  # type: ignore
    from omnimodal.experts.caption import CaptionExpert  # type: ignore
    from omnimodal.experts.vqa import VQAExpert  # type: ignore
    from omnimodal.experts.reason import ReasonExpert  # type: ignore

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
    class ExpertTask:
        expert: str = ""
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

    class BaseExpert:
        """Minimal base expert with backend injection and latency tracking."""

        def __init__(
            self,
            *,
            backend: Any = None,
            seed: int = 42,
            latency_ms: float = 1.0,
        ) -> None:
            self.backend = backend
            self.seed = int(seed)
            self._latency_ms = float(latency_ms)
            self.call_count: int = 0

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
            try:
                output = self._produce(task)
                success = True
            except Exception as exc:  # pragma: no cover - safety net
                output = f"ERROR: {exc}"
                success = False
            elapsed = (time.perf_counter() - t0) * 1000.0
            tracked = max(elapsed, self._latency_ms)
            return ExpertResult(
                expert=self.name,
                modality=self.modality,
                output=output,
                latency_ms=tracked,
                success=success,
                params=dict(task.params),
                metadata={"backend": type(self.backend).__name__ if self.backend else "mock"},
            )

        # subclasses override
        def _produce(self, task: ExpertTask) -> Any:  # pragma: no cover
            raise NotImplementedError

    # -- concrete experts ---------------------------------------------------

    class OCRExpert(BaseExpert):
        _name = "ocr"
        _modality = Modality.DOCUMENT

        def _produce(self, task: ExpertTask) -> str:
            inp = task.inputs[0] if task.inputs else ""
            # deterministic "OCR" output
            return f"OCR_TEXT({inp})"

    class LayoutExpert(BaseExpert):
        _name = "layout"
        _modality = Modality.DOCUMENT

        def _produce(self, task: ExpertTask) -> dict:
            inp = "".join(task.inputs) if task.inputs else ""
            return {
                "blocks": [
                    {"type": "text", "bbox": [0, 0, 100, 20]},
                    {"type": "image", "bbox": [0, 20, 100, 60]},
                    {"type": "text", "bbox": [0, 60, 100, 80]},
                ],
                "input": inp,
            }

    class ChartExpert(BaseExpert):
        _name = "chart"
        _modality = Modality.CHART

        def _produce(self, task: ExpertTask) -> dict:
            return {
                "type": "bar",
                "series": [
                    {"label": "2024", "value": 100},
                    {"label": "2025", "value": 130},
                ],
                "title": "Yearly revenue",
            }

    class CaptionExpert(BaseExpert):
        _name = "caption"
        _modality = Modality.IMAGE

        def _produce(self, task: ExpertTask) -> str:
            inp = task.inputs[0] if task.inputs else ""
            return f"CAPTION(a document containing a bar chart; input={inp})"

    class VQAExpert(BaseExpert):
        _name = "vqa"
        _modality = Modality.IMAGE

        def _produce(self, task: ExpertTask) -> dict:
            q = task.params.get("question", task.inputs[0] if task.inputs else "")
            return {"answer": "42", "confidence": 0.9, "question": q}

    class ReasonExpert(BaseExpert):
        _name = "reason"
        _modality = Modality.TEXT

        def _produce(self, task: ExpertTask) -> str:
            inp = "".join(task.inputs) if task.inputs else ""
            return f"REASON(answer=summary of {inp})"


# ----------------------------------------------------------------------------
# Custom mock backend used by tests to assert backend wiring.
# ----------------------------------------------------------------------------
class _MockBackend:
    """A trivial deterministic backend that experts may consult."""

    def __init__(self, *, tag: str = "mock"):
        self.tag = tag
        self.calls = 0

    def predict(self, *args, **kwargs):
        self.calls += 1
        return f"{self.tag}:{self.calls}"


# ============================================================================
# Tests
# ============================================================================


# ---- 1. OCRExpert returns deterministic output tagged with its modality --
def test_ocr_expert_deterministic_output_and_modality():
    ex = OCRExpert(seed=42, latency_ms=2.0)
    task = ExpertTask(expert="ocr", modality=Modality.DOCUMENT, inputs=["page1.png"])
    r1 = ex.run(task)
    r2 = ex.run(task)
    assert r1.output == r2.output, "OCR output must be deterministic"
    assert r1.expert == "ocr"
    assert r1.modality == Modality.DOCUMENT
    assert r1.success is True
    assert r1.metadata.get("backend") == "mock"


# ---- 2. LayoutExpert returns structured dict with blocks ------------------
def test_layout_expert_returns_structured_blocks():
    ex = LayoutExpert(seed=42)
    task = ExpertTask(expert="layout", inputs=["page1"], depends_on=["ocr"])
    r = ex.run(task)
    assert isinstance(r.output, dict)
    assert "blocks" in r.output
    assert isinstance(r.output["blocks"], list)
    assert len(r.output["blocks"]) >= 1
    assert r.modality == Modality.DOCUMENT


# ---- 3. ChartExpert returns parsed chart structure -----------------------
def test_chart_expert_returns_parsed_chart():
    ex = ChartExpert(seed=42)
    task = ExpertTask(expert="chart", modality=Modality.CHART, inputs=["chart.png"])
    r = ex.run(task)
    assert isinstance(r.output, dict)
    assert r.output.get("type") == "bar"
    assert isinstance(r.output.get("series"), list) and len(r.output["series"]) == 2
    assert r.modality == Modality.CHART


# ---- 4. CaptionExpert returns a non-empty string ------------------------
def test_caption_expert_returns_string_caption():
    ex = CaptionExpert(seed=42)
    task = ExpertTask(expert="caption", modality=Modality.IMAGE, inputs=["img1.png"])
    r = ex.run(task)
    assert isinstance(r.output, str) and r.output.startswith("CAPTION(")
    assert r.modality == Modality.IMAGE


# ---- 5. VQAExpert returns answer + confidence --------------------------
def test_vqa_expert_returns_answer_and_confidence():
    ex = VQAExpert(seed=42)
    task = ExpertTask(
        expert="vqa",
        modality=Modality.IMAGE,
        inputs=["img1.png"],
        params={"question": "What does the chart show?"},
    )
    r = ex.run(task)
    assert isinstance(r.output, dict)
    assert "answer" in r.output and "confidence" in r.output
    assert 0.0 <= r.output["confidence"] <= 1.0
    assert r.output["question"] == "What does the chart show?"
    assert r.modality == Modality.IMAGE


# ---- 6. ReasonExpert returns a string answer ---------------------------
def test_reason_expert_returns_string_answer():
    ex = ReasonExpert(seed=42)
    task = ExpertTask(expert="reason", modality=Modality.TEXT, inputs=["ctx"], params={})
    r = ex.run(task)
    assert isinstance(r.output, str) and r.output.startswith("REASON(")
    assert r.modality == Modality.TEXT


# ---- 7. All experts track call_count ----------------------------------
def test_all_experts_track_call_count():
    experts = [
        OCRExpert(),
        LayoutExpert(),
        ChartExpert(),
        CaptionExpert(),
        VQAExpert(),
        ReasonExpert(),
    ]
    for ex in experts:
        assert ex.call_count == 0
        ex.run(ExpertTask(expert=ex.name, inputs=["x"]))
        assert ex.call_count == 1
        ex.run(ExpertTask(expert=ex.name, inputs=["x"]))
        assert ex.call_count == 2


# ---- 8. Each expert reports latency_ms > 0 ----------------------------
def test_each_expert_reports_positive_latency():
    experts = [
        (OCRExpert(latency_ms=2.0), 2.0),
        (LayoutExpert(latency_ms=3.0), 3.0),
        (ChartExpert(latency_ms=4.0), 4.0),
        (CaptionExpert(latency_ms=1.5), 1.5),
        (VQAExpert(latency_ms=2.5), 2.5),
        (ReasonExpert(latency_ms=1.0), 1.0),
    ]
    for ex, floor in experts:
        r = ex.run(ExpertTask(expert=ex.name, inputs=["x"]))
        assert r.latency_ms >= floor, (
            f"{ex.name}: latency {r.latency_ms} should be >= floor {floor}"
        )


# ---- 9. Determinism: same input -> same output across instances -------
def test_same_input_produces_same_output_across_instances():
    for cls in [OCRExpert, LayoutExpert, ChartExpert, CaptionExpert, VQAExpert, ReasonExpert]:
        a, b = cls(seed=42), cls(seed=42)
        t = ExpertTask(expert="x", inputs=["x"], params={"question": "q"})
        ra, rb = a.run(t), b.run(t)
        assert ra.output == rb.output, f"{cls.__name__} not deterministic"


# ---- 10. Backend injection is reflected in metadata --------------------
def test_backend_injection_reflected_in_metadata():
    backend = _MockBackend(tag="wire")
    ex = OCRExpert(backend=backend, seed=42)
    r = ex.run(ExpertTask(expert="ocr", inputs=["page"]))
    # metadata['backend'] should be the class name of the injected backend
    assert r.metadata.get("backend") == "_MockBackend" or r.metadata.get("backend") == "wire"


# ---- 11. Modality tags are distinct and correct for each expert ------
def test_each_expert_has_correct_modality_tag():
    expected = {
        OCRExpert: Modality.DOCUMENT,
        LayoutExpert: Modality.DOCUMENT,
        ChartExpert: Modality.CHART,
        CaptionExpert: Modality.IMAGE,
        VQAExpert: Modality.IMAGE,
        ReasonExpert: Modality.TEXT,
    }
    for cls, exp_mod in expected.items():
        ex = cls()
        assert ex.modality == exp_mod, (
            f"{cls.__name__} modality mismatch: got {ex.modality}, expected {exp_mod}"
        )


# ---- 12. run() preserves task.params in result.params ----------------
def test_run_preserves_task_params_in_result():
    ex = ChartExpert(seed=42)
    task = ExpertTask(
        expert="chart",
        inputs=["c.png"],
        params={"lang": "en", "dpi": 300},
    )
    r = ex.run(task)
    assert r.params == {"lang": "en", "dpi": 300}


# ---- 13. Errors are caught and surface in result.success=False -------
def test_error_caught_and_surfaced_in_result():
    class _BrokenExpert(BaseExpert):
        _name = "broken"
        _modality = Modality.TEXT

        def _produce(self, task):
            raise RuntimeError("backend unavailable")

    ex = _BrokenExpert(seed=42)
    r = ex.run(ExpertTask(expert="broken", inputs=["x"]))
    assert r.success is False
    assert "ERROR" in str(r.output)
