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
"""OmniFlow modal experts.

Public surface:

* :class:`Expert`, :class:`ExpertRegistry` — expert base class + router.
* :class:`OCRExpert` — text extraction from images / documents.
* :class:`LayoutAnalysisExpert` — document layout region detection.
* :class:`ChartReadingExpert` — chart type detection + data extraction.
* :class:`ImageCaptionExpert` — image captioning.
* :class:`VQAExpert` — visual question answering.
* :class:`TextReasoningExpert` — pure-text reasoning (math / logic / summarization).

All experts work offline with deterministic numpy mock backends. Real OCR /
layout / chart / caption / VQA backends are guarded with try/except and only
loaded when explicitly requested via ``backend=`` and the underlying library
is importable.
"""
from omnimodal.experts.base import (
    ControlToken,
    Expert,
    ExpertRegistry,
    ExpertRequest,
    ExpertResponse,
    ExpertType,
    MemoryEntry,
    ModalityType,
)
from omnimodal.experts.caption import ImageCaptionExpert
from omnimodal.experts.chart import ChartReadingExpert
from omnimodal.experts.layout import LayoutAnalysisExpert
from omnimodal.experts.ocr import OCRExpert
from omnimodal.experts.reason import TextReasoningExpert
from omnimodal.experts.vqa import VQAExpert

__all__ = [
    "Expert",
    "ExpertRegistry",
    "ExpertType",
    "ExpertRequest",
    "ExpertResponse",
    "ModalityType",
    "MemoryEntry",
    "ControlToken",
    "OCRExpert",
    "LayoutAnalysisExpert",
    "ChartReadingExpert",
    "ImageCaptionExpert",
    "VQAExpert",
    "TextReasoningExpert",
]
