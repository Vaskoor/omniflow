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
"""OCRExpert — extracts text from images / documents.

Backend matrix:

==============  =============================  ===============================
``backend``     library (guarded import)        notes
==============  =============================  ===============================
``mock``        (none)                         deterministic numpy fallback
``paddle``      ``paddleocr``                  PaddleOCR (+ angle cls)
``easy``        ``easyocr``                    EasyOCR reader
``tesseract``   ``pytesseract`` + PIL          Tesseract CLI wrapper
==============  =============================  ===============================

When the requested real backend cannot be imported the expert transparently
falls back to the mock backend, ensuring the expert is always usable offline.
"""
from __future__ import annotations

import io
import time
from typing import Any, Optional

import numpy as np

try:  # contract types from om-A
    from omnimodal.core.types import (  # type: ignore[attr-defined]
        ExpertRequest,
        ExpertResponse,
        ExpertType,
        ModalityType,
    )
except ImportError:  # local fallbacks defined in base.py
    from omnimodal.experts.base import (
        ExpertRequest,
        ExpertResponse,
        ExpertType,
        ModalityType,
    )

from omnimodal.experts.base import Expert

# -- guarded real backends --------------------------------------------------
try:
    import paddleocr  # type: ignore
    _HAS_PADDLE = True
except Exception:
    _HAS_PADDLE = False

try:
    import easyocr  # type: ignore
    _HAS_EASY = True
except Exception:
    _HAS_EASY = False

try:
    import pytesseract  # type: ignore
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False


# Word pool for deterministic mock text generation.
_MOCK_WORDS = [
    "page", "title", "introduction", "summary", "conclusion",
    "table", "figure", "data", "result", "method",
    "reference", "abstract", "section", "key", "value",
    "table", "chart", "axis", "label", "scale",
    "row", "column", "cell", "header", "footer",
]


class OCRExpert(Expert):
    """Extract text from images and document pages."""

    def __init__(
        self,
        *,
        backend: str = "mock",
        language: str = "en",
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(backend=backend, seed=seed, **kwargs)
        self.language = language
        self._engine: Any = None
        if backend == "paddle" and _HAS_PADDLE:
            try:
                self._engine = paddleocr.PaddleOCR(use_angle_cls=True, lang=language)
            except Exception:
                self._engine = None
        elif backend == "easy" and _HAS_EASY:
            try:
                self._engine = easyocr.Reader([language])
            except Exception:
                self._engine = None
        elif backend == "tesseract" and _HAS_TESS:
            self._engine = "tesseract"  # sentinel; pytesseract is stateless

    # -- Expert interface ------------------------------------------------
    @property
    def expert_type(self) -> ExpertType:
        return ExpertType.OCR

    @property
    def modalities(self) -> list[ModalityType]:
        return [ModalityType.IMAGE, ModalityType.TEXT]

    async def run(self, request: ExpertRequest) -> ExpertResponse:
        t0 = time.perf_counter()
        try:
            content = request.input_data
            options = dict(request.params or {})
            if self.backend == "mock" or self._engine is None:
                result = self._mock_extract(content, options)
                backend_used = "mock"
            else:
                result = await self._real_extract(content, options)
                backend_used = result.get("backend", self.backend)
            # augment output with backend metadata
            result = {**result, "backend": backend_used, "language": self.language}
            return self._success_response(
                request,
                output=result,
                confidence=float(result.get("confidence", 0.85)),
                latency_ms=self._measure_latency(t0),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._error_response(
                request,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=self._measure_latency(t0),
            )

    # -- mock backend ----------------------------------------------------
    def _mock_extract(self, content: Any, options: dict) -> dict:
        """Deterministic mock OCR — text seeded from the content hash."""
        rng = self._rng_for(content)
        n_lines = int(3 + rng.randint(0, 5))
        lines: list[str] = []
        blocks: list[dict] = []
        for i in range(n_lines):
            n_words = int(3 + rng.randint(0, 6))
            words = rng.choice(_MOCK_WORDS, size=n_words, replace=True).tolist()
            line = " ".join(str(w) for w in words).capitalize()
            lines.append(line)
            conf = float(0.70 + 0.25 * rng.rand())
            blocks.append({
                "text": line,
                "bbox": [0, i * 20, 200, (i + 1) * 20],
                "confidence": round(conf, 4),
            })
        text = "\n".join(lines)
        overall_conf = float(np.mean([b["confidence"] for b in blocks]))
        return {
            "text": text,
            "blocks": blocks,
            "confidence": round(overall_conf, 4),
            "language": self.language,
            "char_count": len(text),
        }

    # -- real backend ----------------------------------------------------
    async def _real_extract(self, content: Any, options: dict) -> dict:
        """Real OCR via paddle / easyocr / tesseract. Guarded."""
        if self.backend == "paddle" and _HAS_PADDLE and self._engine is not None:
            result = self._engine.ocr(content, cls=True)
            lines: list[str] = []
            blocks: list[dict] = []
            for page in result or []:
                for line in page or []:
                    box, (txt, conf) = line
                    lines.append(txt)
                    blocks.append({
                        "text": txt,
                        "bbox": box,
                        "confidence": float(conf),
                    })
            return {
                "text": "\n".join(lines),
                "blocks": blocks,
                "confidence": float(np.mean([b["confidence"] for b in blocks]) if blocks else 0.0),
                "backend": "paddle",
                "language": self.language,
            }
        if self.backend == "easy" and _HAS_EASY and self._engine is not None:
            result = self._engine.readtext(content)
            lines = []
            blocks = []
            for box, txt, conf in result:
                lines.append(txt)
                blocks.append({"text": txt, "bbox": box, "confidence": float(conf)})
            return {
                "text": "\n".join(lines),
                "blocks": blocks,
                "confidence": float(np.mean([b["confidence"] for b in blocks]) if blocks else 0.0),
                "backend": "easy",
                "language": self.language,
            }
        if self.backend == "tesseract" and _HAS_TESS:
            from PIL import Image  # type: ignore

            if isinstance(content, (bytes, bytearray)):
                img = Image.open(io.BytesIO(bytes(content)))
            elif isinstance(content, np.ndarray):
                img = Image.fromarray(content)
            else:
                img = content
            text = pytesseract.image_to_string(img, lang=self.language)
            return {
                "text": text,
                "blocks": [],
                "confidence": 0.85,
                "backend": "tesseract",
                "language": self.language,
            }
        # fall back to mock if real backend unavailable
        return self._mock_extract(content, options)


__all__ = ["OCRExpert"]
