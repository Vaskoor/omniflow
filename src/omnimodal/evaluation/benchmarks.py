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
"""BenchmarkRunner — evaluates the OmniFlow experts + decoding + streaming.

Three benchmarks are provided:

1. **document_understanding** — drives the OCR + layout + chart pipeline over
   mock document samples and checks the structural integrity of the outputs
   (text length, region count, chart type presence).
2. **thinkomni_decoding** — runs the :class:`ThinkOmniDecoder` over
   MathVista-style (image + math question) samples and measures answer
   presence, confidence, and conflict rate.
3. **foresight_streaming** — simulates frame drops over mock frame sequences
   and measures the accuracy + latency impact of carry-forward imputation.

All three have deterministic mock sample generators and work fully offline
(no real OCR / transformers / LLM required).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

try:
    from omnimodal.core.types import (  # type: ignore[attr-defined]
        ExpertRequest,
        ExpertType,
        ModalityType,
    )
except ImportError:
    from omnimodal.experts.base import (
        ExpertRequest,
        ExpertType,
        ModalityType,
    )

from omnimodal.experts.base import ExpertRegistry
from omnimodal.experts.caption import ImageCaptionExpert
from omnimodal.experts.chart import ChartReadingExpert
from omnimodal.experts.layout import LayoutAnalysisExpert
from omnimodal.experts.ocr import OCRExpert
from omnimodal.experts.vqa import VQAExpert

from omnimodal.decoding.thinkomni import (
    MockOmniModalLLM,
    MockTextReasoner,
    ThinkOmniDecoder,
)
from omnimodal.streaming.foresight import (
    ForesightStreamer,
    MockPerceptionStream,
    MockPlanningStream,
    MockSharedEncoder,
)


# ---------------------------------------------------------------------------
# Sample containers
# ---------------------------------------------------------------------------
class SampleSpec(BaseModel):
    """A single benchmark sample."""

    id: str
    kind: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Result of running a single benchmark."""

    name: str
    n_samples: int
    accuracy: float
    avg_latency_ms: float
    std_latency_ms: float
    details: dict[str, Any] = Field(default_factory=dict)
    sample_results: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------
def gen_document_samples(n: int = 10, seed: int = 42) -> list[SampleSpec]:
    """Generate mock document samples for the document_understanding benchmark."""
    rng = np.random.RandomState(seed)
    samples: list[SampleSpec] = []
    chart_types = ["bar", "line", "pie"]
    for i in range(n):
        page_bytes = bytes(rng.randint(0, 255, size=128 + i).tolist())
        has_chart = bool(rng.rand() < 0.5)
        chart_type = str(rng.choice(chart_types)) if has_chart else None
        samples.append(SampleSpec(
            id=f"doc_{i:04d}",
            kind="document",
            inputs={
                "image_bytes": page_bytes,
                "has_chart": has_chart,
                "chart_type": chart_type,
            },
            expected={
                "min_ocr_chars": 1,
                "min_layout_regions": 1,
                "expected_chart_type": chart_type,
            },
        ))
    return samples


def gen_mathvista_samples(n: int = 10, seed: int = 42) -> list[SampleSpec]:
    """Generate mock MathVista-style (image + math question) samples."""
    rng = np.random.RandomState(seed)
    samples: list[SampleSpec] = []
    questions = [
        ("What is 2 + 3?", "5"),
        ("What is 7 - 4?", "3"),
        ("What is 6 * 5?", "30"),
        ("What is 9 / 3?", "3"),
        ("What is 2 ^ 4?", "16"),
        ("What is 100 - 25?", "75"),
        ("What is 8 * 8?", "64"),
        ("What is 1 + 1 + 1?", "3"),
        ("What is 10 / 5?", "2"),
        ("What is 12 * 12?", "144"),
        ("What is 4 + 9?", "13"),
        ("What is 17 - 8?", "9"),
    ]
    for i in range(n):
        q, expected = questions[i % len(questions)]
        image_bytes = bytes(rng.randint(0, 255, size=64 + i).tolist())
        samples.append(SampleSpec(
            id=f"mathvista_{i:04d}",
            kind="mathvista",
            inputs={
                "query": q,
                "image_bytes": image_bytes,
            },
            expected={
                "answer": expected,
                "any_text": True,
            },
        ))
    return samples


def gen_foresight_frame_drop_samples(n: int = 10, seed: int = 42) -> list[SampleSpec]:
    """Generate mock frame-sequence samples with drop patterns."""
    rng = np.random.RandomState(seed)
    samples: list[SampleSpec] = []
    for i in range(n):
        n_frames = int(5 + rng.randint(0, 10))
        frames = [bytes(rng.randint(0, 255, size=32).tolist()) for _ in range(n_frames)]
        n_drop = min(n_frames - 1, 1 + int(rng.randint(0, 3)))
        drop_idx = sorted(rng.choice(n_frames, size=n_drop, replace=False).tolist()) if n_drop > 0 else []
        samples.append(SampleSpec(
            id=f"foresight_{i:04d}",
            kind="foresight",
            inputs={
                "frames": frames,
                "n_frames": n_frames,
                "drop_indices": [int(x) for x in drop_idx],
            },
            expected={
                "n_frames": n_frames,
                "drop_count": len(drop_idx),
                "max_accuracy_delta": 1.0,
            },
        ))
    return samples


def _is_success(response: Any) -> bool:
    """Heuristic: an expert response is successful when ``confidence > 0``
    and ``output`` is non-None (the contract has no ``success`` flag)."""
    if response is None:
        return False
    conf = getattr(response, "confidence", 0.0) or 0.0
    out = getattr(response, "output", None)
    # error responses carry {"error": ...}
    if isinstance(out, dict) and "error" in out:
        return False
    return conf > 0.0 and out is not None


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------
class BenchmarkRunner:
    """Evaluates OmniFlow on three benchmarks.

    Parameters
    ----------
    registry
        :class:`ExpertRegistry` with the experts wired in. If ``None``, a
        default registry with mock-backend experts is constructed.
    decoder
        :class:`ThinkOmniDecoder` instance. If ``None``, one with mock
        text-reasoner and omni-modal-LLM is constructed.
    streamer
        :class:`ForesightStreamer` instance. If ``None``, one with mock
        streams and shared encoder is constructed.
    seed
        Determinism seed for sample generation.
    """

    BENCHMARK_NAMES = (
        "document_understanding",
        "thinkomni_decoding",
        "foresight_streaming",
    )

    def __init__(
        self,
        *,
        registry: Optional[ExpertRegistry] = None,
        decoder: Optional[ThinkOmniDecoder] = None,
        streamer: Optional[ForesightStreamer] = None,
        seed: int = 42,
    ):
        self.seed = seed
        self.registry = registry or ExpertRegistry(experts=[
            OCRExpert(backend="mock", seed=seed),
            LayoutAnalysisExpert(backend="mock", seed=seed),
            ChartReadingExpert(backend="mock", seed=seed),
            ImageCaptionExpert(backend="mock", seed=seed),
            VQAExpert(backend="mock", seed=seed),
        ])
        self.decoder = decoder or ThinkOmniDecoder(
            MockTextReasoner(vocab_size=64, seed=seed),
            MockOmniModalLLM(vocab_size=64, seed=seed),
            fusion_strategy="adaptive",
            seed=seed,
        )
        self.streamer = streamer or ForesightStreamer(
            MockPerceptionStream(seed=seed),
            MockPlanningStream(seed=seed),
            MockSharedEncoder(feature_dim=64, seed=seed),
            seed=seed,
        )

    # ------------------------------------------------------------------
    async def run_benchmark(self, name: str, n_samples: int = 10) -> BenchmarkResult:
        """Run one of the three benchmarks."""
        if name not in self.BENCHMARK_NAMES:
            raise ValueError(
                f"unknown benchmark {name!r}; expected one of {self.BENCHMARK_NAMES}"
            )
        if name == "document_understanding":
            return await self._bench_document(n_samples)
        if name == "thinkomni_decoding":
            return await self._bench_thinkomni(n_samples)
        return await self._bench_foresight(n_samples)

    # ------------------------------------------------------------------
    # Document understanding (OCR + layout + chart pipeline)
    # ------------------------------------------------------------------
    async def _bench_document(self, n_samples: int) -> BenchmarkResult:
        samples = gen_document_samples(n_samples, seed=self.seed)
        per_sample: list[dict[str, Any]] = []
        accuracies: list[float] = []
        latencies: list[float] = []
        for sample in samples:
            t0 = time.perf_counter()
            image_bytes = sample.inputs["image_bytes"]
            expected = sample.expected
            # run OCR
            ocr_req = ExpertRequest(
                input_data=image_bytes,
                modality=ModalityType.IMAGE,
                params={"task": "extract"},
            )
            ocr_resp = await self.registry.call(ExpertType.OCR, ocr_req)
            # run layout
            layout_req = ExpertRequest(
                input_data=image_bytes,
                modality=ModalityType.LAYOUT,
                params={"page_width": 800, "page_height": 1100},
            )
            layout_resp = await self.registry.call(ExpertType.LAYOUT_ANALYSIS, layout_req)
            # run chart if expected
            chart_type = None
            if expected.get("expected_chart_type"):
                chart_req = ExpertRequest(
                    input_data=image_bytes,
                    modality=ModalityType.CHART,
                )
                chart_resp = await self.registry.call(ExpertType.CHART_READING, chart_req)
                if _is_success(chart_resp) and isinstance(chart_resp.output, dict):
                    chart_type = chart_resp.output.get("chart_type")
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            ocr_out = ocr_resp.output if isinstance(ocr_resp.output, dict) else {}
            layout_out = layout_resp.output if isinstance(layout_resp.output, dict) else {}
            ocr_ok = (
                _is_success(ocr_resp)
                and len(ocr_out.get("text", "")) >= expected.get("min_ocr_chars", 0)
            )
            layout_ok = (
                _is_success(layout_resp)
                and len(layout_out.get("regions", [])) >= expected.get("min_layout_regions", 0)
            )
            chart_ok = True
            if expected.get("expected_chart_type"):
                chart_ok = chart_type == expected["expected_chart_type"]
                weight = 3
                score = sum([ocr_ok, layout_ok, chart_ok]) / weight
            else:
                weight = 2
                score = sum([ocr_ok, layout_ok]) / weight
            accuracies.append(float(score))
            per_sample.append({
                "id": sample.id,
                "ocr_ok": ocr_ok,
                "layout_ok": layout_ok,
                "chart_ok": chart_ok,
                "chart_type_detected": chart_type,
                "latency_ms": lat,
                "score": round(score, 4),
            })
        return BenchmarkResult(
            name="document_understanding",
            n_samples=len(samples),
            accuracy=float(np.mean(accuracies)) if accuracies else 0.0,
            avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            std_latency_ms=float(np.std(latencies)) if latencies else 0.0,
            details={
                "experts_called": ["OCR", "LayoutAnalysis", "ChartReading"],
                "has_chart_samples": sum(1 for s in samples if s.inputs.get("has_chart")),
            },
            sample_results=per_sample,
        )

    # ------------------------------------------------------------------
    # ThinkOmni collaborative decoding (MathVista-style)
    # ------------------------------------------------------------------
    async def _bench_thinkomni(self, n_samples: int) -> BenchmarkResult:
        samples = gen_mathvista_samples(n_samples, seed=self.seed)
        per_sample: list[dict[str, Any]] = []
        accuracies: list[float] = []
        latencies: list[float] = []
        conflict_count = 0
        for sample in samples:
            t0 = time.perf_counter()
            query = sample.inputs["query"]
            image = sample.inputs["image_bytes"]
            answer = await self.decoder.decode(query=query, image=image, context=None)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            expected_answer = sample.expected.get("answer", "")
            hit = (
                expected_answer.lower() in answer.lower()
                if expected_answer else bool(answer)
            )
            accuracies.append(1.0 if hit else 0.0)
            diag = self.decoder.last_diagnostics()
            if diag.get("conflict", {}).get("has_conflict"):
                conflict_count += 1
            per_sample.append({
                "id": sample.id,
                "query": query,
                "expected": expected_answer,
                "answer": answer,
                "hit": hit,
                "alpha": diag.get("alpha"),
                "conflict": diag.get("conflict", {}).get("has_conflict", False),
                "latency_ms": lat,
            })
        return BenchmarkResult(
            name="thinkomni_decoding",
            n_samples=len(samples),
            accuracy=float(np.mean(accuracies)) if accuracies else 0.0,
            avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            std_latency_ms=float(np.std(latencies)) if latencies else 0.0,
            details={
                "fusion_strategy": self.decoder.fusion_strategy,
                "conflict_rate": conflict_count / max(1, len(samples)),
                "hit_count": sum(accuracies),
            },
            sample_results=per_sample,
        )

    # ------------------------------------------------------------------
    # Foresight streaming (frame-drop robustness)
    # ------------------------------------------------------------------
    async def _bench_foresight(self, n_samples: int) -> BenchmarkResult:
        samples = gen_foresight_frame_drop_samples(n_samples, seed=self.seed)
        per_sample: list[dict[str, Any]] = []
        accuracies: list[float] = []
        latencies: list[float] = []
        max_deltas: list[float] = []
        for sample in samples:
            t0 = time.perf_counter()
            frames = sample.inputs["frames"]
            drop_indices = sample.inputs["drop_indices"]
            self.streamer.reset()
            for idx, frame in enumerate(frames):
                await self.streamer.process_frame(frame, idx)
            impact = self.streamer.handle_frame_drop(drop_indices)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            max_delta = float(impact.get("max_accuracy_delta", 0.0))
            max_deltas.append(max_delta)
            score = max(0.0, 1.0 - max_delta)
            accuracies.append(score)
            per_sample.append({
                "id": sample.id,
                "n_frames": len(frames),
                "drop_count": len(drop_indices),
                "drop_indices": drop_indices,
                "avg_accuracy_delta": impact.get("avg_accuracy_delta"),
                "max_accuracy_delta": impact.get("max_accuracy_delta"),
                "avg_latency_ms": impact.get("avg_latency_ms"),
                "robustness_score": round(score, 4),
                "total_latency_ms": lat,
            })
        return BenchmarkResult(
            name="foresight_streaming",
            n_samples=len(samples),
            accuracy=float(np.mean(accuracies)) if accuracies else 0.0,
            avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            std_latency_ms=float(np.std(latencies)) if latencies else 0.0,
            details={
                "avg_max_delta": float(np.mean(max_deltas)) if max_deltas else 0.0,
                "frames_processed_total": sum(s.inputs["n_frames"] for s in samples),
                "frames_dropped_total": sum(len(s.inputs["drop_indices"]) for s in samples),
            },
            sample_results=per_sample,
        )

    # ------------------------------------------------------------------
    async def run_all(self, n_samples: int = 10) -> list[BenchmarkResult]:
        """Run all three benchmarks sequentially."""
        results: list[BenchmarkResult] = []
        for name in self.BENCHMARK_NAMES:
            results.append(await self.run_benchmark(name, n_samples=n_samples))
        return results


__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "SampleSpec",
    "gen_document_samples",
    "gen_mathvista_samples",
    "gen_foresight_frame_drop_samples",
]


# ---------------------------------------------------------------------------
# Console-script entry point — `omniflow-benchmark`
# ---------------------------------------------------------------------------
def main() -> None:
    """``omniflow-benchmark`` console-script entry point.

    Runs all three benchmarks (document understanding, ThinkOmni decoding,
    Foresight streaming) against deterministic mock backends, then prints a
    compact summary table + JSON dump to stdout.  No external services are
    required — the whole sweep finishes in well under a second.
    """
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        prog="omniflow-benchmark",
        description="Run the OmniFlow offline benchmark sweep.",
    )
    parser.add_argument(
        "-n", "--n-samples", type=int, default=10,
        help="Number of samples per benchmark category (default: 10).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print machine-readable JSON instead of a human table.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Determinism seed for sample generation (default: 42).",
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(seed=args.seed)
    results = asyncio.run(runner.run_all(n_samples=args.n_samples))

    if args.json:
        payload = [r.model_dump(mode="json") for r in results]
        print(_json.dumps({"benchmarks": payload}, indent=2, default=str))
        return

    # Human-readable summary table.
    print(
        "\n  OmniFlow v0.1.0 — offline benchmark sweep\n"
        "  ─────────────────────────────────────────────────────────────────────\n"
        f"  samples per category: {args.n_samples}    seed: {args.seed}\n"
        "  ─────────────────────────────────────────────────────────────────────\n"
    )
    print(f"  {'benchmark':<26} {'accuracy':>10} {'avg ms':>10} {'std ms':>10} {'n':>4}")
    print("  " + "-" * 64)
    for r in results:
        print(
            f"  {r.name:<26} "
            f"{r.accuracy * 100:>9.1f}% "
            f"{r.avg_latency_ms:>10.3f} "
            f"{r.std_latency_ms:>10.3f} "
            f"{r.n_samples:>4}"
        )
    print(
        "\n  all three benchmarks finished successfully — the framework runs\n"
        "  end-to-end against deterministic mock backends, no PyTorch /\n"
        "  transformers / LLM API keys required.\n"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
