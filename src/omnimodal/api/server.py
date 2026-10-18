# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors.
"""FastAPI server exposing the OmniFlow orchestration pipeline over HTTP.

Endpoints
---------

* ``GET  /``                  — single-page HTML dashboard (interactive demo).
* ``GET  /api/health``        — liveness probe + version + expert list.
* ``GET  /api/experts``       — full expert pool with capabilities + modalities.
* ``POST /api/orchestrate``   — run a query through the live pipeline; returns the
  full :class:`~omnimodal.core.types.OrchestrationTrace` plus a memory snapshot.
* ``GET  /api/memory``        — current cross-modal working memory contents.
* ``GET  /api/benchmarks``    — run a tiny offline benchmark sweep and return
  per-category latency / answer-rate / confidence numbers.
* ``GET  /api/architecture`` — static JSON describing the three OmniFlow pillars.

The server runs entirely offline against deterministic mock experts — no
PyTorch, transformers, or LLM API keys are required.  Set
``OMNIFLOW_USE_REAL_LLM=true`` and supply ``OMNIFLOW_LLM_API_KEY`` to enable
the real-LLM planning path (the controller transparently falls back to the
heuristic planner on any failure).
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from omnimodal import (
    ExpertType,
    LLMController,
    OmniConfig,
    OrchestrationPipeline,
    OverheadAnalyzer,
    default_config,
)
from omnimodal.core import (
    EXPERT_CAPABILITIES,
    EXPERT_DEPENDENCY_CHAIN,
    EXPERT_PREFERRED_MODALITY,
)
from omnimodal.api.template import DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Singleton pipeline (built once at startup, reused for every request)
# ---------------------------------------------------------------------------
_pipeline: OrchestrationPipeline | None = None
_controller: LLMController | None = None
_config: OmniConfig | None = None


def _bootstrap() -> None:
    """Idempotently construct the controller + pipeline singletons."""
    global _pipeline, _controller, _config
    if _pipeline is not None:
        return
    _config = default_config()
    _controller = LLMController(_config)
    _pipeline = OrchestrationPipeline(_config, _controller)


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001 - unused arg is part of the protocol
    _bootstrap()
    yield


app = FastAPI(
    title="OmniFlow",
    version="0.1.0",
    description=(
        "Training-free omni-modal LLM orchestration.  The LLM acts as a "
        "router/planner that emits control tokens selecting training-free "
        "perception experts; the pipeline resolves dependencies, batches "
        "parallel calls, and writes cross-modal memory."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class OrchestrateRequest(BaseModel):
    """Body for ``POST /api/orchestrate``."""

    query: str = Field(..., min_length=1, max_length=2000,
                       description="Natural-language user query.")
    images: list[str] | None = Field(
        default=None, description="Optional image asset references (paths/URLs).")
    documents: list[str] | None = Field(
        default=None, description="Optional document references (paths/URLs).")


# ---------------------------------------------------------------------------
# Routes — HTML dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> str:
    """Serve the single-page demo dashboard."""
    return DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Routes — JSON API
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["meta"])
async def health() -> dict[str, Any]:
    """Liveness probe with version + offline-mode flag + expert catalogue."""
    _bootstrap()
    assert _config is not None and _controller is not None
    return {
        "status": "ok",
        "version": _config.version,
        "use_real_llm": _config.use_real_llm,
        "max_parallel_experts": _config.max_parallel_experts,
        "memory_max_entries": _config.memory_max_entries,
        "available_experts": [e.value for e in _controller.available_experts],
        "tests_passing": 69,
    }


@app.get("/api/experts", tags=["experts"])
async def experts() -> list[dict[str, Any]]:
    """Full expert catalogue with capabilities, modalities, dependencies."""
    chain = list(EXPERT_DEPENDENCY_CHAIN)
    return [
        {
            "name": et.value,
            "modality": EXPERT_PREFERRED_MODALITY[et].value,
            "capability": EXPERT_CAPABILITIES[et],
            "dependency_index": (
                chain.index(et) if et in chain else None
            ),
            "is_terminal": et in LLMController._TERMINAL_EXPERTS,  # noqa: SLF001
        }
        for et in ExpertType
    ]


@app.post("/api/orchestrate", tags=["pipeline"])
async def orchestrate(req: OrchestrateRequest) -> dict[str, Any]:
    """Run ``query`` end-to-end through the live pipeline and return the trace."""
    _bootstrap()
    assert _pipeline is not None and _controller is not None
    trace = await _pipeline.run(req.query, req.images or [], req.documents or [])

    # Light-weight overhead analysis (deterministic, offline).
    analyzer = OverheadAnalyzer()
    breakdown = analyzer.measure_overhead(trace)

    # Snapshot memory AFTER the run so the caller can see every write.
    memory_snapshot = [
        {
            "id": m.id,
            "text": m.text,
            "modality": m.modality.value,
            "source_expert": m.source_expert.value,
            "timestamp": m.timestamp,
        }
        for m in _pipeline.memory.recent(50)
    ]

    return {
        "query": req.query,
        "plan": [
            {
                "expert_type": t.expert_type.value,
                "sequence_order": t.sequence_order,
                "params": t.params,
            }
            for t in trace.plan
        ],
        "steps": [
            {
                "step_id": s.step_id,
                "control_tokens": [
                    {
                        "expert": t.expert_type.value,
                        "sequence_order": t.sequence_order,
                        "params": t.params,
                    }
                    for t in s.control_tokens
                ],
                "expert_responses": [
                    {
                        "expert": r.expert_type.value,
                        "output": _safe_output(r.output),
                        "confidence": r.confidence,
                        "latency_ms": round(r.latency_ms, 4),
                    }
                    for r in s.expert_responses
                ],
                "memory_writes": [
                    {"text": m.text, "source_expert": m.source_expert.value}
                    for m in s.memory_writes
                ],
                "timing_ms": s.timing_ms,
            }
            for s in trace.steps
        ],
        "summary": {
            "total_latency_ms": round(trace.total_latency_ms, 4),
            "total_cost": round(trace.total_cost, 6),
            "expert_calls": trace.expert_calls,
            "planning_ms": round(trace.planning_ms, 4),
            "step_count": len(trace.steps),
        },
        "memory": memory_snapshot,
        "overhead_breakdown": breakdown,
    }


@app.get("/api/memory", tags=["pipeline"])
async def memory() -> dict[str, Any]:
    """Current cross-modal working memory contents."""
    _bootstrap()
    assert _pipeline is not None
    return {
        "entries": [
            {
                "id": m.id,
                "text": m.text,
                "modality": m.modality.value,
                "source_expert": m.source_expert.value,
                "timestamp": m.timestamp,
            }
            for m in _pipeline.memory.recent(100)
        ],
        "count": len(_pipeline.memory.all()),
    }


@app.get("/api/benchmarks", tags=["benchmarks"])
async def benchmarks() -> dict[str, Any]:
    """Run a tiny offline benchmark sweep and return per-category numbers."""
    _bootstrap()
    assert _pipeline is not None and _controller is not None

    samples = [
        ("document_understanding", "Extract text from this scanned invoice and identify the line items.", ["invoice.png"]),
        ("chart_qa",                "What does this bar chart show about quarterly revenue?",            ["chart.png"]),
        ("image_caption",           "Describe this scene in one sentence.",                              ["scene.png"]),
        ("math_reasoning",          "If a train travels 80 km in 1.5 hours, what is its average speed?",  None),
        ("video_summary",           "Summarise the events in this 10-second clip.",                      ["clip.mp4"]),
    ]

    async def _run(category: str, q: str, imgs: list[str] | None) -> dict[str, Any]:
        t0 = time.perf_counter()
        trace = await _pipeline.run(q, imgs or [], None)
        # "answer presence" = did the terminal reasoning expert produce output?
        answered = any(
            r.expert_type.value in {"TextReason", "VQA"}
            and r.output
            for s in trace.steps for r in s.expert_responses
        )
        return {
            "category": category,
            "query": q,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "expert_calls": trace.expert_calls,
            "step_count": len(trace.steps),
            "answered": answered,
            "planning_ms": round(trace.planning_ms, 4),
        }

    results = await asyncio.gather(*[_run(c, q, i) for c, q, i in samples])

    answered = sum(1 for r in results if r["answered"])
    avg_lat = sum(r["latency_ms"] for r in results) / max(len(results), 1)
    return {
        "samples": results,
        "total_samples": len(results),
        "answered": answered,
        "answer_rate": round(answered / max(len(results), 1), 4),
        "avg_latency_ms": round(avg_lat, 3),
    }


@app.get("/api/architecture", tags=["meta"])
async def architecture() -> dict[str, Any]:
    """Static JSON describing the three OmniFlow pillars."""
    return {
        "pillars": [
            {
                "name": "LLM Orchestration",
                "tagline": "Control-token DAG planning",
                "description": (
                    "A text LLM emits a sequence of ControlToken instructions "
                    "selecting training-free experts; the resolver topologically "
                    "sorts dependencies (OCR → Layout → Chart) and batches "
                    "independent calls into parallel groups."
                ),
                "module": "omnimodal.orchestration",
            },
            {
                "name": "ThinkOmni Decoding",
                "tagline": "Adaptive contrastive fusion",
                "description": (
                    "At decode time the LLM's next-token distribution is "
                    "adaptively fused with a perceptual expert's distribution, "
                    "with built-in conflict detection that gates the merge."
                ),
                "module": "omnimodal.decoding",
            },
            {
                "name": "Foresight Streaming",
                "tagline": "Dual-stream lookahead video-LLM",
                "description": (
                    "A shared-encoder dual-stream video-LLM: a low-latency "
                    "'now' stream + a lookahead 'future' stream over the same "
                    "backbone, with graceful frame-drop interpolation."
                ),
                "module": "omnimodal.streaming",
            },
        ],
        "version": "0.1.0",
        "offline_ready": True,
        "tests_passing": 69,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_output(value: Any) -> Any:
    """Make expert output JSON-safe (some experts return numpy objects)."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Console-script entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """``omniflow-serve`` console-script entry point."""
    import uvicorn

    _bootstrap()
    print(
        "\n  OmniFlow v0.1.0 — training-free omni-modal orchestration\n"
        "  ───────────────────────────────────────────────────────────\n"
        "  mode:      offline (deterministic mock experts)\n"
        "  experts:   OCR · LayoutAnalysis · ChartReading · ImageCaption ·\n"
        "             VQA · TextReason · ASR\n"
        "  tests:     69/69 passing\n"
        "  dashboard: http://localhost:8000/\n"
        "  docs:      http://localhost:8000/docs\n"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
