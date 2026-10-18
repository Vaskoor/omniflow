# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.

"""``omnimodal.orchestration`` — LLM-driven planning + execution + overhead.

This package wires the three orchestration roles together:

* :class:`LLMController` — the planner.  Turns a query + context into a list of
  :class:`~omnimodal.core.types.ControlToken` instructions.
* :class:`OrchestrationPipeline` — the executor.  Consumes a control-token
  plan, dispatches experts in parallel/sequential stages, and writes
  cross-modal memory, producing an :class:`~omnimodal.core.types.OrchestrationTrace`.
* :class:`OverheadAnalyzer` — the accountant.  Decomposes a trace into
  orchestration overhead and compares it against a trained baseline.
"""

from __future__ import annotations

from omnimodal.orchestration.controller import (
    LLMCallable,
    LLMController,
    PlannerContext,
)
from omnimodal.orchestration.overhead import OverheadAnalyzer
from omnimodal.orchestration.pipeline import (
    CrossModalMemory,
    ExpertProtocol,
    ExpertRegistry,
    MockExpert,
    OrchestrationPipeline,
)

__all__ = [
    "LLMController",
    "LLMCallable",
    "PlannerContext",
    "OrchestrationPipeline",
    "ExpertProtocol",
    "ExpertRegistry",
    "MockExpert",
    "CrossModalMemory",
    "OverheadAnalyzer",
]
