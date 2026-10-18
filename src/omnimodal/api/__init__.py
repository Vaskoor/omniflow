# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors
"""``omnimodal.api`` — FastAPI server + single-page demo dashboard.

Exposes the live orchestration pipeline over HTTP and serves a polished
single-page HTML dashboard at ``/`` so anyone can run a query end-to-end
in the browser without writing Python.

Public surface:

* :obj:`app` — the configured :class:`fastapi.FastAPI` instance.
* :func:`main` — the ``omniflow-serve`` console-script entry point.

Run locally::

    omniflow-serve
    # or
    python -m omnimodal.api.server

Then open ``http://localhost:8000/``.
"""
from __future__ import annotations

from omnimodal.api.server import app, main

__all__ = ["app", "main"]
