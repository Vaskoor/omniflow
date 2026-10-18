# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# Top-level package.  Re-exports the core contract and the orchestration
# entry points so users can write::
#
#     from omnimodal import (
#         OmniConfig, LLMController, OrchestrationPipeline, OverheadAnalyzer,
#         ControlToken, ExpertType, OrchestrationTrace,
#     )
#
# Optional sub-packages (``experts``, ``memory``, ``decoding``, ``streaming``,
# ``evaluation``) are imported lazily by their own users; this module never
# imports them, so a partially-shipped tree still loads cleanly.

"""OmniFlow — unified training-free omni-modal LLM orchestration.

OmniFlow combines three ideas into one training-free stack:

* **LLM Orchestration** — a text LLM acts as a router/planner that emits a
  sequence of :class:`~omnimodal.core.types.ControlToken` instructions
  selecting + ordering training-free expert modules.
* **ThinkOmni collaborative decoding** — (in :mod:`omnimodal.decoding`) expert
  logits are merged at decode time rather than at the prompt level.
* **Foresight streaming planning** — (in :mod:`omnimodal.streaming`) plans are
  emitted incrementally so the first token of the answer can stream before the
  whole plan is settled.

This module exposes only the *core + orchestration* surface.  Heavy optional
dependencies (the LLM SDK, the expert backends, etc.) are never imported here.
"""

from __future__ import annotations

from omnimodal.core import (
    ControlToken,
    ExpertRequest,
    ExpertResponse,
    ExpertType,
    MemoryEntry,
    ModalityType,
    OmniConfig,
    OrchestrationStep,
    OrchestrationTrace,
    default_config,
    load_config,
)
from omnimodal.orchestration import (
    LLMController,
    OrchestrationPipeline,
    OverheadAnalyzer,
)

#: Package version (PEP 440).
__version__: str = "0.1.0"

__all__ = [
    "__version__",
    # core contract
    "ModalityType",
    "ExpertType",
    "ControlToken",
    "ExpertRequest",
    "ExpertResponse",
    "MemoryEntry",
    "OrchestrationStep",
    "OrchestrationTrace",
    "OmniConfig",
    "load_config",
    "default_config",
    # orchestration surface
    "LLMController",
    "OrchestrationPipeline",
    "OverheadAnalyzer",
]
