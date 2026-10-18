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
"""ImageCaptionExpert — generates natural-language captions for images.

Backends:

==============  ===========================  ==============================
``backend``     library (guarded)            notes
==============  ===========================  ==============================
``mock``        (none)                       template-based, deterministic
``blip``        ``transformers`` BLIP        Salesforce/blip-image-captioning-base
``llava``       ``transformers`` LLaVA       llava-hf/llava-1.5-7b-hf (or 1b)
==============  ===========================  ==============================
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
    import transformers  # type: ignore
    _HAS_HF = True
except Exception:
    _HAS_HF = False
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# Caption object pool for the template-based mock backend.
_OBJECTS = [
    "a cat", "a dog", "a person", "a car", "a building", "a tree",
    "a book", "a clock", "a chair", "a table", "a window", "a sign",
    "a chart", "a graph", "a document", "a page", "a screen", "a plate",
]
_SCENES = [
    "indoors", "outdoors", "in an office", "in a park", "at night",
    "during the day", "in a kitchen", "on a street", "in a classroom",
]
_ACTIONS = [
    "sitting", "standing", "running", "reading", "writing", "looking",
    "displaying data", "showing a trend", "lying flat", "being viewed",
]
_QUALITIES = [
    "well-lit", "blurry", "in focus", "high-resolution", "noisy",
    "colored", "grayscale", "annotated",
]


class ImageCaptionExpert(Expert):
    """Generate natural-language captions for images."""

    def __init__(
        self,
        *,
        backend: str = "mock",
        model_name: Optional[str] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(backend=backend, seed=seed, **kwargs)
        self.model_name = model_name or {
            "blip": "Salesforce/blip-image-captioning-base",
            "llava": "llava-hf/llava-1.5-7b-hf",
        }.get(backend, None)
        self._processor: Any = None
        self._model: Any = None
        if backend in {"blip", "llava"} and _HAS_HF and _HAS_TORCH:
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq  # type: ignore
                self._processor = AutoProcessor.from_pretrained(self.model_name)
                self._model = AutoModelForVision2Seq.from_pretrained(self.model_name)
            except Exception:
                self._processor = None
                self._model = None

    # -- Expert interface ------------------------------------------------
    @property
    def expert_type(self) -> ExpertType:
        return ExpertType.IMAGE_CAPTION

    @property
    def modalities(self) -> list[ModalityType]:
        return [ModalityType.IMAGE, ModalityType.TEXT]

    async def run(self, request: ExpertRequest) -> ExpertResponse:
        t0 = time.perf_counter()
        try:
            content = request.input_data
            params = dict(request.params or {})
            query = params.get("query")  # optional prompt for conditional captioning
            if self.backend == "mock" or self._model is None:
                caption = self._mock_caption(content)
                backend_used = "mock"
            else:
                caption = self._real_caption(content, query)
                backend_used = self.backend
            output = {
                "caption": caption,
                "backend": backend_used,
                "caption_length": len(caption),
            }
            return self._success_response(
                request,
                output=output,
                confidence=float(0.7 + 0.25 * self._rng_for(content).rand()),
                latency_ms=self._measure_latency(t0),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._error_response(
                request,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=self._measure_latency(t0),
            )

    # -- mock backend ----------------------------------------------------
    def _mock_caption(self, content: Any) -> str:
        """Template-based caption seeded deterministically from the content hash."""
        rng = self._rng_for(content)
        obj = str(rng.choice(_OBJECTS))
        scene = str(rng.choice(_SCENES))
        action = str(rng.choice(_ACTIONS))
        quality = str(rng.choice(_QUALITIES))
        templates = [
            f"A {quality} image of {obj} {action}, {scene}.",
            f"This image shows {obj} {action} {scene}; the photo is {quality}.",
            f"Photograph of {obj} {action}; scene: {scene}; quality: {quality}.",
            f"{obj.capitalize()} {action} {scene}. The image is {quality}.",
        ]
        return str(rng.choice(templates))

    # -- real backend ----------------------------------------------------
    def _real_caption(self, content: Any, query: Optional[str]) -> str:
        """Real captioning via BLIP / LLaVA."""
        try:
            from PIL import Image  # type: ignore
            import io as _io
            if isinstance(content, (bytes, bytearray)):
                img = Image.open(_io.BytesIO(bytes(content))).convert("RGB")
            elif isinstance(content, np.ndarray):
                img = Image.fromarray(content)
            else:
                img = content
            if self.backend == "blip":
                inputs = self._processor(images=img, return_tensors="pt")
                out = self._model.generate(**inputs, max_new_tokens=50)
                return self._processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            if self.backend == "llava":
                prompt = f"USER: <image>\nDescribe this image.{(' ' + query) if query else ''}\nASSISTANT:"
                inputs = self._processor(text=prompt, images=img, return_tensors="pt")
                out = self._model.generate(**inputs, max_new_tokens=100)
                text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
                return text.split("ASSISTANT:")[-1].strip()
        except Exception:
            return self._mock_caption(content)
        return self._mock_caption(content)


__all__ = ["ImageCaptionExpert"]
