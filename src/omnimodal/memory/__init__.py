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
"""OmniFlow cross-modal memory store.

Currently exposes :class:`CrossModalMemory` — a text-centric store with
TF-IDF / keyword-overlap retrieval (pure numpy) and an optional
sentence-transformers backend for real dense embeddings.
"""
from omnimodal.memory.cross_modal import CrossModalMemory

__all__ = ["CrossModalMemory"]
