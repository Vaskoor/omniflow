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
"""OmniFlow benchmarks.

Exposes :class:`BenchmarkRunner` and the auxiliary :class:`BenchmarkResult`.
"""
from omnimodal.evaluation.benchmarks import (
    BenchmarkResult,
    BenchmarkRunner,
    SampleSpec,
    gen_document_samples,
    gen_foresight_frame_drop_samples,
    gen_mathvista_samples,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "SampleSpec",
    "gen_document_samples",
    "gen_mathvista_samples",
    "gen_foresight_frame_drop_samples",
]
