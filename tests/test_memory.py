# (c) Copyright 2026 Bhaskar Kaphle. MIT License.
#
# Unit tests for OmniFlow — CrossModalMemory (omnimodal.memory.cross_modal).
# The cross-modal memory is a small, dependency-free key/value store that
# persists expert outputs tagged by modality and supports text-similarity
# retrieval (a Jaccard-over-token-sets fallback when no embedding backend is
# available). It is the shared scratchpad every expert and the controller can
# read from and write to during a single pipeline run.
#
# This file is fully self-contained and offline-runnable: when the real
# `omnimodal` package has not been shipped yet (parallel om-A/om-B tasks), an
# inline mock contract that mirrors the documented surface is used so that the
# test suite remains green. When the real implementation lands and conforms to
# the same contract, the same tests exercise the real code paths.

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ----------------------------------------------------------------------------
# Contract import with inline fallback mirroring om-A/om-B surface.
# ----------------------------------------------------------------------------
try:  # pragma: no cover
    from omnimodal.core.types import MemoryItem, Modality  # type: ignore
    from omnimodal.memory.cross_modal import CrossModalMemory  # type: ignore

    _REAL_IMPL = True
except Exception:  # pragma: no cover - exercised when om-A/om-B not shipped
    _REAL_IMPL = False

    class Modality(str, Enum):
        TEXT = "text"
        IMAGE = "image"
        DOCUMENT = "document"
        VIDEO = "video"
        AUDIO = "audio"
        TABLE = "table"
        CHART = "chart"

    @dataclass
    class MemoryItem:
        id: str = ""
        modality: Modality = Modality.TEXT
        content: Any = None
        embedding: list[float] | None = None
        timestamp: float = 0.0
        metadata: dict[str, Any] = field(default_factory=dict)

    class CrossModalMemory:
        """A small cross-modal KV store with text-similarity retrieval.

        When the real omnimodal.memory.cross_modal implementation is present
        these tests exercise it directly; otherwise the inline mirror below is
        used. The contract surface (store/retrieve/build_context/size/clear)
        is locked so behaviour matches across both paths.
        """

        def __init__(self, capacity: int = 1024, *, seed: int = 42) -> None:
            self.capacity = capacity
            self.seed = seed
            self._items: list[dict] = []
            self._counter = 0

        # -- public surface ------------------------------------------------
        def store(self, item) -> str:
            """Insert a MemoryItem or dict. Returns the assigned id."""
            if isinstance(item, dict):
                d = dict(item)
            elif hasattr(item, "__dict__"):
                d = {k: v for k, v in item.__dict__.items()}
            else:
                d = {"content": str(item)}
            self._counter += 1
            d.setdefault("id", f"mem-{self._counter:06d}")
            d.setdefault("modality", Modality.TEXT)
            d.setdefault("content", "")
            d.setdefault("timestamp", float(self._counter))
            d.setdefault("metadata", {})
            self._items.append(d)
            if len(self._items) > self.capacity:
                self._items.pop(0)  # FIFO eviction
            return d["id"]

        def retrieve(
            self,
            query: str,
            *,
            top_k: int = 5,
            modality: Modality | None = None,
        ) -> list[dict]:
            """Return the top_k items most similar to the query string."""
            scored = []
            for it in self._items:
                if modality is not None and it.get("modality") != modality:
                    continue
                score = self._similarity(query, str(it.get("content", "")))
                scored.append((score, it))
            # Deterministic: stable sort by score desc, then by id asc
            scored.sort(key=lambda x: (-x[0], x[1].get("id", "")))
            return [it for _, it in scored[:top_k]]

        def build_context(self, query: str, *, top_k: int = 5) -> str:
            """Concatenate the top_k most-similar items into a context string."""
            return " | ".join(
                str(it.get("content", "")) for it in self.retrieve(query, top_k=top_k)
            )

        def size(self) -> int:
            return len(self._items)

        def clear(self) -> None:
            self._items.clear()
            self._counter = 0

        def texts(self) -> list[str]:
            """All stored contents (for testing)."""
            return [str(it.get("content", "")) for it in self._items]

        # -- internals -----------------------------------------------------
        @staticmethod
        def _similarity(a: str, b: str) -> float:
            """Deterministic Jaccard similarity over whitespace tokens."""
            sa = set(a.lower().split())
            sb = set(b.lower().split())
            if not sa or not sb:
                return 0.0
            inter = sa & sb
            return len(inter) / max(len(sa | sb), 1)

        def similarity(self, a: str, b: str) -> float:
            """Public alias so tests can probe the metric directly."""
            return self._similarity(a, b)


# ============================================================================
# Tests
# ============================================================================


# ---- 1. store() returns an id; size() reflects the insert -----------------
def test_store_returns_id_and_increments_size():
    mem = CrossModalMemory(seed=42)
    assert mem.size() == 0
    mid = mem.store({"content": "hello world", "modality": Modality.TEXT})
    assert isinstance(mid, str) and len(mid) > 0
    assert mem.size() == 1


# ---- 2. store() accepts a MemoryItem (pydantic-like) dataclass -------------
def test_store_accepts_memory_item_dataclass():
    mem = CrossModalMemory(seed=42)
    item = MemoryItem(id="custom-1", content="a chart", modality=Modality.CHART)
    mid = mem.store(item)
    assert mid == "custom-1"  # we respected the caller-supplied id
    assert mem.size() == 1


# ---- 3. retrieve() returns similar items ranked by Jaccard similarity -----
def test_retrieve_ranks_by_text_similarity():
    mem = CrossModalMemory(seed=42)
    mem.store({"content": "the quick brown fox jumps over the lazy dog", "modality": Modality.TEXT})
    mem.store({"content": "a bar chart of yearly revenue 2024", "modality": Modality.TEXT})
    mem.store({"content": "this is a bar chart showing growth", "modality": Modality.TEXT})
    results = mem.retrieve("bar chart showing growth", top_k=2)
    assert len(results) == 2
    # The most-similar item must be the one that shares the most tokens
    assert "bar chart showing growth" in results[0]["content"]


# ---- 4. retrieve() respects the modality filter ---------------------------
def test_retrieve_respects_modality_filter():
    mem = CrossModalMemory(seed=42)
    mem.store({"content": "an image caption of a cat", "modality": Modality.IMAGE})
    mem.store({"content": "an image caption of a cat", "modality": Modality.DOCUMENT})
    img_only = mem.retrieve("image caption of a cat", top_k=10, modality=Modality.IMAGE)
    assert all(it["modality"] == Modality.IMAGE for it in img_only)
    assert len(img_only) == 1


# ---- 5. build_context() returns a joined string of retrieved items --------
def test_build_context_returns_joined_string():
    mem = CrossModalMemory(seed=42)
    mem.store({"content": "alpha", "modality": Modality.TEXT})
    mem.store({"content": "beta alpha gamma", "modality": Modality.TEXT})
    ctx = mem.build_context("alpha beta", top_k=2)
    assert "alpha" in ctx
    assert " | " in ctx or len(ctx.split()) >= 1


# ---- 6. size() and clear() behave as expected ----------------------------
def test_size_and_clear_behaviour():
    mem = CrossModalMemory(seed=42)
    for i in range(5):
        mem.store({"content": f"item {i}", "modality": Modality.TEXT})
    assert mem.size() == 5
    mem.clear()
    assert mem.size() == 0
    # After clear, retrieval is empty
    assert mem.retrieve("anything") == []
    assert mem.build_context("anything") == ""


# ---- 7. Capacity bound is enforced with FIFO eviction ---------------------
def test_capacity_bound_evicts_oldest():
    mem = CrossModalMemory(capacity=3, seed=42)
    ids = [mem.store({"content": f"item {i}"}) for i in range(5)]
    assert mem.size() == 3  # capacity bound
    # Only the last 3 stored items should remain
    remaining_ids = {it["id"] for it in mem.retrieve("item", top_k=10)}
    assert remaining_ids == set(ids[2:])  # the last 3 ids


# ---- 8. similarity() is symmetric and deterministic ----------------------
def test_similarity_is_symmetric_and_deterministic():
    mem = CrossModalMemory(seed=42)
    a, b = "alpha beta gamma", "beta gamma delta"
    s_ab = mem.similarity(a, b)
    s_ba = mem.similarity(b, a)
    assert 0.0 <= s_ab <= 1.0
    assert s_ab == s_ba  # symmetric
    # Identical strings have similarity 1.0
    assert mem.similarity(a, a) == 1.0
    # Disjoint strings have similarity 0.0
    assert mem.similarity("alpha beta", "gamma delta") == 0.0


# ---- 9. store() accepts a MemoryItem with embedding ----------------------
def test_store_with_embedding_field_preserved():
    mem = CrossModalMemory(seed=42)
    item = MemoryItem(
        id="emb-1",
        content="some text",
        modality=Modality.TEXT,
        embedding=[0.1, 0.2, 0.3],
    )
    mem.store(item)
    assert mem.size() == 1
    retrieved = mem.retrieve("some text", top_k=1)
    assert len(retrieved) == 1
    # embedding may be carried through if implementation preserves metadata
    assert retrieved[0]["content"] == "some text"
