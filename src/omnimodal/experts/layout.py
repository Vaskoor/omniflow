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
"""LayoutAnalysisExpert — detects document layout regions.

Layout regions are classified into one of:

* ``text`` — paragraph / line block
* ``title`` — heading or title
* ``table`` — tabular region
* ``figure`` — image / figure / chart
* ``header`` / ``footer`` — page header / footer
* ``list`` — bulleted or numbered list

Backends:

==============  ============================  ==================================
``backend``     library (guarded)             notes
==============  ============================  ==================================
``mock``        (none)                        deterministic numpy fallback
``layoutparser`` ``layoutparser`` + model   LayoutParser pre-trained detector
``detectron2``  ``detectron2``                Custom Detectron2 model required
==============  ============================  ==================================
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
    import layoutparser  # type: ignore
    _HAS_LP = True
except Exception:
    _HAS_LP = False

try:
    import detectron2  # type: ignore
    _HAS_DET2 = True
except Exception:
    _HAS_DET2 = False


# Region type pool for mock backend (deterministic).
_REGION_TYPES = ["text", "title", "table", "figure", "header", "footer", "list"]


class LayoutAnalysisExpert(Expert):
    """Detects structural regions in document images / pages."""

    def __init__(
        self,
        *,
        backend: str = "mock",
        model_name: Optional[str] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(backend=backend, seed=seed, **kwargs)
        self.model_name = model_name
        self._engine: Any = None
        if backend == "layoutparser" and _HAS_LP and model_name:
            try:
                self._engine = layoutparser.models.detectron2.layoutparser_model(
                    model_name, device="cpu"
                )
            except Exception:
                self._engine = None
        elif backend == "detectron2" and _HAS_DET2:
            # User must supply a configured predictor via kwargs['predictor'].
            self._engine = kwargs.get("predictor")

    # -- Expert interface ------------------------------------------------
    @property
    def expert_type(self) -> ExpertType:
        return ExpertType.LAYOUT_ANALYSIS

    @property
    def modalities(self) -> list[ModalityType]:
        return [ModalityType.IMAGE, ModalityType.LAYOUT]

    async def run(self, request: ExpertRequest) -> ExpertResponse:
        t0 = time.perf_counter()
        try:
            content = request.input_data
            options = dict(request.params or {})
            page_w = int(options.get("page_width", 1000))
            page_h = int(options.get("page_height", 1400))
            if self.backend == "mock" or self._engine is None:
                regions = self._mock_layout(content, page_w, page_h)
                backend_used = "mock"
            else:
                regions = await self._real_layout(content)
                backend_used = self.backend
            output = {
                "regions": regions,
                "page": {"width": page_w, "height": page_h},
                "n_regions": len(regions),
                "backend": backend_used,
            }
            return self._success_response(
                request,
                output=output,
                confidence=float(np.mean([r["confidence"] for r in regions]) if regions else 0.0),
                latency_ms=self._measure_latency(t0),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._error_response(
                request,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=self._measure_latency(t0),
            )

    # -- mock backend ----------------------------------------------------
    def _mock_layout(self, content: Any, page_w: int, page_h: int) -> list[dict]:
        """Deterministic mock layout seeded from the content hash."""
        rng = self._rng_for(content)
        n_regions = int(3 + rng.randint(0, 7))
        regions: list[dict] = []
        ys = sorted(rng.randint(0, max(1, page_h - 100), size=n_regions).tolist())
        for i, y in enumerate(ys):
            rtype = str(rng.choice(_REGION_TYPES))
            h = int(40 + rng.randint(0, 200))
            w = int(page_w - rng.randint(0, 200))
            x = int(rng.randint(0, 100))
            region = {
                "id": f"r{i:03d}",
                "type": rtype,
                "bbox": [int(x), int(y), int(x + w), int(min(page_h, y + h))],
                "confidence": round(float(0.70 + 0.25 * rng.rand()), 4),
                "text": "",
            }
            if rtype in {"text", "title", "header", "footer", "list"}:
                n_words = int(3 + rng.randint(0, 6))
                words = rng.choice(
                    ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
                     "adipiscing", "elit", "sed", "do", "eiusmod"],
                    size=n_words, replace=True,
                ).tolist()
                region["text"] = " ".join(str(w) for w in words)
            elif rtype == "table":
                rows = int(1 + rng.randint(0, 4))
                cols = int(2 + rng.randint(0, 3))
                region["table_shape"] = [rows, cols]
            elif rtype == "figure":
                region["caption"] = "Figure " + str(i + 1)
            regions.append(region)
        return regions

    # -- real backend ----------------------------------------------------
    async def _real_layout(self, content: Any) -> list[dict]:
        """Real layout detection via layoutparser / detectron2."""
        if self.backend == "layoutparser" and _HAS_LP and self._engine is not None:
            import layoutparser as lp  # type: ignore
            layout = self._engine.detect(content)
            regions: list[dict] = []
            for i, blk in enumerate(layout):
                x1, y1, x2, y2 = blk.coordinates
                regions.append({
                    "id": f"r{i:03d}",
                    "type": str(blk.type).lower() if blk.type else "text",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(blk.score),
                    "text": "",
                })
            return regions
        if self.backend == "detectron2" and _HAS_DET2 and self._engine is not None:
            outputs = self._engine(content)
            instances = outputs["instances"].to("cpu")
            regions = []
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                x1, y1, x2, y2 = box.tolist()
                regions.append({
                    "id": f"r{i:03d}",
                    "type": _REGION_TYPES[int(cls) % len(_REGION_TYPES)],
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(score),
                    "text": "",
                })
            return regions
        # fall back to mock
        return self._mock_layout(content, 1000, 1400)


__all__ = ["LayoutAnalysisExpert"]
