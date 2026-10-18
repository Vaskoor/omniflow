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
"""ChartReadingExpert — extracts structured data from chart images.

Supported chart types: ``bar``, ``line``, ``pie``. The mock backend derives
chart type and series data deterministically from the content hash. Real
backend hooks (``opencv``-based pixel segmentation or a transformers
chart-parsing model) are guarded so the package works offline.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

try:
    from omnimodal.core.types import (  # type: ignore[attr-defined]
        ExpertRequest,
        ExpertResponse,
        ExpertType,
        ModalityType,
    )
except ImportError:
    from omnimodal.experts.base import (
        ExpertRequest,
        ExpertResponse,
        ExpertType,
        ModalityType,
    )

from omnimodal.experts.base import Expert

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import transformers  # type: ignore
    _HAS_HF = True
except Exception:
    _HAS_HF = False


_CHART_TYPES = ["bar", "line", "pie"]
_SERIES_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_LABEL_POOL = [
    "Q1", "Q2", "Q3", "Q4", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "A", "B", "C", "D",
    "2020", "2021", "2022", "2023", "2024", "Alpha", "Beta", "Gamma", "Delta",
]


class ChartReadingExpert(Expert):
    """Detects chart type and extracts structured data from chart images."""

    def __init__(
        self,
        *,
        backend: str = "mock",
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(backend=backend, seed=seed, **kwargs)
        self._engine: Any = None
        if backend == "opencv" and _HAS_CV2:
            self._engine = cv2
        elif backend == "transformers" and _HAS_HF:
            try:
                # Defer actual model load to call site; transformers chart
                # parsing is not a single canonical model. Allow user-supplied
                # parser via kwargs['parser'].
                self._engine = kwargs.get("parser")
            except Exception:
                self._engine = None

    # -- Expert interface ------------------------------------------------
    @property
    def expert_type(self) -> ExpertType:
        return ExpertType.CHART_READING

    @property
    def modalities(self) -> list[ModalityType]:
        return [ModalityType.IMAGE, ModalityType.CHART]

    async def run(self, request: ExpertRequest) -> ExpertResponse:
        t0 = time.perf_counter()
        try:
            content = request.input_data
            if self.backend == "mock" or self._engine is None:
                result = self._mock_extract(content)
                backend_used = "mock"
            else:
                result = self._real_extract(content)
                backend_used = self.backend
            result = {**result, "backend": backend_used}
            return self._success_response(
                request,
                output=result,
                confidence=float(result.get("confidence", 0.8)),
                latency_ms=self._measure_latency(t0),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._error_response(
                request,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=self._measure_latency(t0),
            )

    # -- mock backend ----------------------------------------------------
    def _mock_extract(self, content: Any) -> dict:
        """Deterministic mock chart data seeded from the content hash."""
        rng = self._rng_for(content)
        chart_type = str(rng.choice(_CHART_TYPES))
        n_series = int(1 + rng.randint(0, 3))
        n_points = int(3 + rng.randint(0, 8))
        x_labels = [str(x) for x in rng.choice(_LABEL_POOL, size=n_points, replace=False).tolist()]
        series = []
        for s in range(n_series):
            name = f"Series {s + 1}"
            color = _SERIES_COLORS[s % len(_SERIES_COLORS)]
            if chart_type == "pie":
                values = [float(round(rng.rand() * 100, 2)) for _ in range(min(n_points, 5))]
                total = sum(values) or 1.0
                values = [round(v / total * 100, 2) for v in values]
            else:
                values = [float(round(rng.rand() * 100, 2)) for _ in range(n_points)]
            series.append({
                "name": name,
                "color": color,
                "values": values,
            })
        x_title = "Category"
        y_title = "Value" if chart_type != "pie" else "Percent"
        title = f"{chart_type.capitalize()} chart of {y_title.lower()} over {x_title.lower()}"
        return {
            "chart_type": chart_type,
            "title": title,
            "x_axis": {"title": x_title, "labels": x_labels},
            "y_axis": {"title": y_title},
            "series": series,
            "legend": [s["name"] for s in series],
            "confidence": round(float(0.75 + 0.20 * rng.rand()), 4),
        }

    # -- real backend ----------------------------------------------------
    def _real_extract(self, content: Any) -> dict:
        """Real chart extraction via opencv / transformers. Guarded."""
        if self.backend == "opencv" and _HAS_CV2 and self._engine is not None:
            try:
                if isinstance(content, (bytes, bytearray)):
                    arr = np.frombuffer(content, dtype=np.uint8)
                    img = self._engine.imdecode(arr, self._engine.IMREAD_COLOR)
                elif isinstance(content, np.ndarray):
                    img = content
                else:
                    img = content
                gray = self._engine.cvtColor(img, self._engine.COLOR_BGR2GRAY)
                _, thresh = self._engine.threshold(gray, 240, 255, self._engine.THRESH_BINARY_INV)
                contours, _ = self._engine.findContours(thresh, self._engine.RETR_EXTERNAL, self._engine.CHAIN_APPROX_SIMPLE)
                bars = sorted([self._engine.boundingRect(c) for c in contours], key=lambda b: b[0])
                series_values = [float(b[3]) for b in bars]  # heights
                return {
                    "chart_type": "bar",
                    "title": "auto-detected bar chart",
                    "x_axis": {"title": "Category", "labels": [f"b{i}" for i in range(len(bars))]},
                    "y_axis": {"title": "Value"},
                    "series": [{"name": "Series 1", "color": _SERIES_COLORS[0], "values": series_values}],
                    "legend": ["Series 1"],
                    "confidence": 0.7,
                }
            except Exception:
                return self._mock_extract(content)
        if self.backend == "transformers" and _HAS_HF and self._engine is not None:
            try:
                # User-supplied parser callable: parser(image) -> dict
                return {**self._engine(content)}
            except Exception:
                return self._mock_extract(content)
        # fall back to mock
        return self._mock_extract(content)


__all__ = ["ChartReadingExpert"]
