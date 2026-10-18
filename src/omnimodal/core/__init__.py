# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.

"""``omnimodal.core`` — public type contract and configuration.

This package re-exports the full contract surface declared in
:mod:`omnimodal.core.types` plus the configuration helpers from
:mod:`omnimodal.core.config`, so downstream code can do::

    from omnimodal.core import (
        ModalityType, ExpertType, ControlToken, ExpertRequest,
        ExpertResponse, MemoryEntry, OrchestrationStep, OrchestrationTrace,
        OmniConfig, load_config,
    )

without reaching into sub-modules.
"""

from __future__ import annotations

from omnimodal.core.config import default_config, load_config
from omnimodal.core.types import (
    EXPERT_CAPABILITIES,
    EXPERT_DEPENDENCY_CHAIN,
    EXPERT_PREFERRED_MODALITY,
    ControlToken,
    ExpertRequest,
    ExpertResponse,
    ExpertType,
    MemoryEntry,
    ModalityType,
    OmniConfig,
    OrchestrationStep,
    OrchestrationTrace,
)

__all__ = [
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
    "EXPERT_CAPABILITIES",
    "EXPERT_DEPENDENCY_CHAIN",
    "EXPERT_PREFERRED_MODALITY",
]
