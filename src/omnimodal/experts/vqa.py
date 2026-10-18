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
"""VQAExpert — visual question answering.

Given an image and a natural-language question, returns an answer.

Backends:

==============  ===========================  ==============================
``backend``     library (guarded)            notes
==============  ===========================  ==============================
``mock``        (none)                       keyword-based deterministic
``vilt``        ``transformers`` ViLT        dandelin/vilt-b32-finetuned-vqa
``llava``       ``transformers`` LLaVA       llava-hf/llava-1.5-7b-hf
==============  ===========================  ==============================

The question is supplied via ``request.params['query']`` (the contract's
``ExpertRequest`` does not carry a top-level ``query`` field).
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


# Keyword -> answer templates for the mock backend. The mock returns an answer
# deterministically chosen from this table based on the question keywords.
_ANSWER_TABLE: dict[str, list[str]] = {
    "color": ["blue", "red", "green", "yellow", "black", "white"],
    "how many": ["three", "two", "five", "one", "four"],
    "count": ["three", "two", "five", "one", "four"],
    "what": ["a chart", "a document", "an image", "a table", "a figure"],
    "where": ["top-left", "centre", "bottom-right", "top-right", "bottom-left"],
    "when": ["today", "yesterday", "this year", "morning", "afternoon"],
    "who": ["the author", "the user", "a researcher", "an analyst"],
    "is": ["yes", "no", "yes", "no"],
    "are": ["yes", "no"],
    "do": ["yes", "no"],
    "does": ["yes", "no"],
    "can you": ["yes", "no"],
    "type": ["bar", "line", "pie", "table", "text"],
}
_DEFAULT_ANSWERS = ["I cannot tell from the image.", "It is unclear.", "Not visible."]


class VQAExpert(Expert):
    """Answer natural-language questions about an image."""

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
            "vilt": "dandelin/vilt-b32-finetuned-vqa",
            "llava": "llava-hf/llava-1.5-7b-hf",
        }.get(backend, None)
        self._processor: Any = None
        self._model: Any = None
        if backend in {"vilt", "llava"} and _HAS_HF and _HAS_TORCH:
            try:
                from transformers import (  # type: ignore
                    ViltProcessor,
                    ViltForQuestionAnswering,
                    AutoProcessor,
                    LlavaForConditionalGeneration,
                )
                if backend == "vilt":
                    self._processor = ViltProcessor.from_pretrained(self.model_name)
                    self._model = ViltForQuestionAnswering.from_pretrained(self.model_name)
                else:
                    self._processor = AutoProcessor.from_pretrained(self.model_name)
                    self._model = LlavaForConditionalGeneration.from_pretrained(self.model_name)
            except Exception:
                self._processor = None
                self._model = None

    # -- Expert interface ------------------------------------------------
    @property
    def expert_type(self) -> ExpertType:
        return ExpertType.VQA

    @property
    def modalities(self) -> list[ModalityType]:
        return [ModalityType.IMAGE, ModalityType.TEXT]

    async def run(self, request: ExpertRequest) -> ExpertResponse:
        t0 = time.perf_counter()
        try:
            content = request.input_data
            params = dict(request.params or {})
            question = params.get("query") or params.get("question") or "What is in this image?"
            if self.backend == "mock" or self._model is None:
                answer, confidence = self._mock_answer(content, question)
                backend_used = "mock"
            else:
                answer, confidence = self._real_answer(content, question)
                backend_used = self.backend
            output = {
                "answer": answer,
                "question": question,
                "backend": backend_used,
            }
            return self._success_response(
                request,
                output=output,
                confidence=confidence,
                latency_ms=self._measure_latency(t0),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._error_response(
                request,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=self._measure_latency(t0),
            )

    # -- mock backend ----------------------------------------------------
    def _mock_answer(self, content: Any, question: str) -> tuple[str, float]:
        """Keyword-based deterministic mock answer."""
        q_lower = question.lower()
        rng = self._rng_for((content, question))
        for key, options in _ANSWER_TABLE.items():
            if key in q_lower:
                ans = str(rng.choice(options))
                return ans, float(0.6 + 0.3 * rng.rand())
        ans = str(rng.choice(_DEFAULT_ANSWERS))
        return ans, float(0.3 + 0.2 * rng.rand())

    # -- real backend ----------------------------------------------------
    def _real_answer(self, content: Any, question: str) -> tuple[str, float]:
        """Real VQA via ViLT / LLaVA. Guarded."""
        try:
            from PIL import Image  # type: ignore
            import io as _io
            if isinstance(content, (bytes, bytearray)):
                img = Image.open(_io.BytesIO(bytes(content))).convert("RGB")
            elif isinstance(content, np.ndarray):
                img = Image.fromarray(content)
            else:
                img = content
            if self.backend == "vilt":
                inputs = self._processor(images=img, text=question, return_tensors="pt")
                out = self._model(**inputs)
                logits = out.logits
                idx = int(logits.argmax(-1).item())
                ans = self._model.config.id2label[idx]
                conf = float(torch.softmax(logits, dim=-1).max().item())
                return ans, conf
            if self.backend == "llava":
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                inputs = self._processor(text=prompt, images=img, return_tensors="pt")
                out = self._model.generate(**inputs, max_new_tokens=80)
                text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
                ans = text.split("ASSISTANT:")[-1].strip()
                return ans, 0.85
        except Exception:
            return self._mock_answer(content, question)
        return self._mock_answer(content, question)


__all__ = ["VQAExpert"]
