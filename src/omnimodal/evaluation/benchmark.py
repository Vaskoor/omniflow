# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors.
"""``omnimodal.evaluation.benchmark`` — singular-name shim.

The actual implementation lives in :mod:`omnimodal.evaluation.benchmarks`
(plural).  This module exists so the ``omniflow-benchmark`` console-script
entry point declared in ``pyproject.toml`` resolves cleanly to
``omnimodal.evaluation.benchmark:main`` without renaming the file.
"""
from __future__ import annotations

from omnimodal.evaluation.benchmarks import (
    BenchmarkResult,
    BenchmarkRunner,
    SampleSpec,
    gen_document_samples,
    gen_foresight_frame_drop_samples,
    gen_mathvista_samples,
    main,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "SampleSpec",
    "gen_document_samples",
    "gen_mathvista_samples",
    "gen_foresight_frame_drop_samples",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
