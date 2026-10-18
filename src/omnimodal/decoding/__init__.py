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
"""OmniFlow training-free collaborative decoding.

Exposes :class:`ThinkOmniDecoder` and the auxiliary :class:`ConflictReport`.
"""
from omnimodal.decoding.thinkomni import (
    ConflictReport,
    ConflictType,
    MockOmniModalLLM,
    MockTextReasoner,
    OmniModalLLMProtocol,
    TextReasonerProtocol,
    ThinkOmniDecoder,
    adaptive_weighted_contrastive_scaling,
)

__all__ = [
    "ThinkOmniDecoder",
    "ConflictReport",
    "ConflictType",
    "TextReasonerProtocol",
    "OmniModalLLMProtocol",
    "MockTextReasoner",
    "MockOmniModalLLM",
    "adaptive_weighted_contrastive_scaling",
]
