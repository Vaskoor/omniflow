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
"""OmniFlow dual-stream Foresight streaming.

Exposes :class:`ForesightStreamer` plus the auxiliary :class:`StreamOutput`
and :class:`PlanAction` containers.
"""
from omnimodal.streaming.foresight import (
    ActionType,
    ForesightStreamer,
    MockPerceptionStream,
    MockPlanningStream,
    MockSharedEncoder,
    PerceptionStreamProtocol,
    PlanningStreamProtocol,
    PlanAction,
    SharedEncoderProtocol,
    StreamOutput,
)

__all__ = [
    "ForesightStreamer",
    "StreamOutput",
    "PlanAction",
    "ActionType",
    "PerceptionStreamProtocol",
    "PlanningStreamProtocol",
    "SharedEncoderProtocol",
    "MockPerceptionStream",
    "MockPlanningStream",
    "MockSharedEncoder",
]
