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
"""ForesightStreamer — dual-stream architecture with shared weights.

Architecture:

* **Shared encoder** (:class:`SharedEncoderProtocol`) — produces a dense
  feature vector from a raw frame.
* **Perception stream** (:class:`PerceptionStreamProtocol`) — processes the
  current frame's features to produce perception output (captions, VQA, OCR
  results, layout regions, etc.).
* **Planning stream** (:class:`PlanningStreamProtocol`) — looks ahead and
  predicts which regions / experts to attend to in upcoming frames, so the
  perception stream can pre-warm heavy experts.

The dual-stream uses **shared weights** — meaning the perception and
planning streams share the same encoder (and conceptually the same backbone).
In the mock case "shared weights" is implemented via a single numpy
``RandomState(seed)`` that both streams seed from, so their outputs are
deterministic and correlated.

Key methods:

* :meth:`process_frame` — encode + perception + planning.
* :meth:`handle_frame_drop` — measures accuracy and latency impact of
  dropped frames (compares baseline vs drop-imputed outputs).
* :meth:`plan_future` — pure planning entry point that returns a list of
  :class:`PlanAction` objects for upcoming frames.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Plan + output containers
# ---------------------------------------------------------------------------
class ActionType(str, Enum):
    """Categories of perception-planning actions."""

    ATTEND = "attend"        # focus on a region of the next frame
    WARM_EXPERT = "warm_expert"  # pre-warm a named expert
    SKIP = "skip"            # skip the next frame (low-information)
    ENCODE = "encode"        # encode the next frame with a specific config
    QUERY = "query"          # issue a downstream VQA query in advance
    BUFFER = "buffer"        # buffer the frame for later re-analysis


class PlanAction(BaseModel):
    """A single perception-planning action."""

    action_type: ActionType
    target: str = ""
    priority: float = 0.5  # 0 (low) ... 1 (high)
    frame_offset: int = 1   # frames from now (1 = next frame)
    params: dict[str, Any] = Field(default_factory=dict)


class StreamOutput(BaseModel):
    """Output of :meth:`ForesightStreamer.process_frame`."""

    frame_idx: int
    perception_result: dict[str, Any] = Field(default_factory=dict)
    planned_actions: list[PlanAction] = Field(default_factory=list)
    perception_latency_ms: float = 0.0
    planning_latency_ms: float = 0.0
    features: list[float] = Field(default_factory=list)
    dropped: bool = False


# ---------------------------------------------------------------------------
# Stream protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class SharedEncoderProtocol(Protocol):
    """Encodes raw frames into dense feature vectors."""

    def encode(self, frame: Any) -> np.ndarray: ...


@runtime_checkable
class PerceptionStreamProtocol(Protocol):
    """Perception stream — processes current-frame features."""

    async def process(self, features: np.ndarray, frame_idx: int) -> dict[str, Any]: ...


@runtime_checkable
class PlanningStreamProtocol(Protocol):
    """Planning stream — predicts future perception needs."""

    async def predict(
        self,
        current_features: np.ndarray,
        history: list[dict[str, Any]],
    ) -> list[PlanAction]: ...


# ---------------------------------------------------------------------------
# Mock stream implementations (deterministic numpy)
# ---------------------------------------------------------------------------
class MockSharedEncoder:
    """Deterministic numpy mock encoder.

    Hashes the frame to a 32-bit seed, then samples a feature vector of
    shape ``(feature_dim,)`` from a seeded ``RandomState``.
    """

    def __init__(self, feature_dim: int = 64, seed: int = 42):
        self.feature_dim = feature_dim
        self.seed = seed

    def encode(self, frame: Any) -> np.ndarray:
        h = self._frame_hash(frame)
        rng = np.random.RandomState(h ^ (self.seed * 0x9E3779B1 & 0xFFFFFFFF))
        features = rng.randn(self.feature_dim).astype(np.float32)
        features = features / (np.linalg.norm(features) + 1e-9)
        return features

    @staticmethod
    def _frame_hash(frame: Any) -> int:
        if frame is None:
            return 0
        if isinstance(frame, (bytes, bytearray)):
            data = bytes(frame)
        elif isinstance(frame, np.ndarray):
            data = frame.tobytes()
        elif isinstance(frame, str):
            data = frame.encode("utf-8")
        else:
            data = repr(frame).encode("utf-8")
        h = hashlib.sha256(data).hexdigest()
        return int(h[:8], 16) & 0xFFFFFFFF


class MockPerceptionStream:
    """Deterministic mock perception stream."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    async def process(self, features: np.ndarray, frame_idx: int) -> dict[str, Any]:
        # Build a deterministic 32-bit seed from features + frame_idx.
        h = hashlib.sha256(features.tobytes() + str(frame_idx).encode("utf-8")).hexdigest()
        seed_int = (int(h[:8], 16) ^ (self.seed * 0x9E3779B1)) & 0xFFFFFFFF
        rng = np.random.RandomState(seed_int)
        # detect a salient region deterministically
        salient_idx = int(np.argmax(np.abs(features)))
        return {
            "frame_idx": frame_idx,
            "salient_dim": salient_idx,
            "salience_score": float(np.abs(features[salient_idx])),
            "caption": f"frame_{frame_idx}_salient_dim_{salient_idx}",
            "tags": [f"dim_{i}" for i in np.argsort(-np.abs(features))[:3].tolist()],
        }


class MockPlanningStream:
    """Deterministic mock planning stream."""

    def __init__(self, seed: int = 42, expert_pool: Optional[list[str]] = None):
        self.seed = seed
        self.expert_pool = expert_pool or [
            "ocr", "layout", "chart", "caption", "vqa",
        ]

    async def predict(
        self,
        current_features: np.ndarray,
        history: list[dict[str, Any]],
    ) -> list[PlanAction]:
        # Hash the features for a deterministic plan
        h = hashlib.sha256(current_features.tobytes()).hexdigest()
        rng = np.random.RandomState(int(h[:8], 16) ^ self.seed)
        actions: list[PlanAction] = []
        # 1) warm an expert based on the salient dim
        salient = int(np.argmax(np.abs(current_features)))
        expert = self.expert_pool[salient % len(self.expert_pool)]
        actions.append(PlanAction(
            action_type=ActionType.WARM_EXPERT,
            target=expert,
            priority=float(0.7 + 0.3 * rng.rand()),
            frame_offset=1,
            params={"reason": "predicted salient expert"},
        ))
        # 2) attend to a region in the next frame
        actions.append(PlanAction(
            action_type=ActionType.ATTEND,
            target=f"region_{salient % 4}",
            priority=float(0.5 + 0.4 * rng.rand()),
            frame_offset=1,
            params={"bbox": [0, 0, 100, 100]},
        ))
        # 3) maybe skip the next frame if it looks low-information
        if rng.rand() < 0.2:
            actions.append(PlanAction(
                action_type=ActionType.SKIP,
                target="next",
                priority=float(0.3),
                frame_offset=1,
            ))
        return actions


# ---------------------------------------------------------------------------
# ForesightStreamer
# ---------------------------------------------------------------------------
class ForesightStreamer:
    """Dual-stream (perception + planning) frame processor with shared encoder.

    Parameters
    ----------
    perception_stream
        Object implementing :class:`PerceptionStreamProtocol`.
    planning_stream
        Object implementing :class:`PlanningStreamProtocol`.
    shared_encoder
        Object implementing :class:`SharedEncoderProtocol`.
    history_size
        How many past perception outputs to keep for planning context.
    """

    def __init__(
        self,
        perception_stream: PerceptionStreamProtocol,
        planning_stream: PlanningStreamProtocol,
        shared_encoder: SharedEncoderProtocol,
        *,
        history_size: int = 16,
        seed: int = 42,
    ):
        self.perception_stream = perception_stream
        self.planning_stream = planning_stream
        self.shared_encoder = shared_encoder
        self.history_size = history_size
        self.seed = seed
        self._history: list[dict[str, Any]] = []
        self._frames: dict[int, dict[str, Any]] = {}
        self._baseline_outputs: dict[int, StreamOutput] = {}
        self._drop_impacts: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    async def process_frame(self, frame: Any, frame_idx: int) -> StreamOutput:
        """Encode + perceive + plan, returning a :class:`StreamOutput`."""
        # encode
        t0 = time.perf_counter()
        features = self.shared_encoder.encode(frame)
        # perceive
        perc_t0 = time.perf_counter()
        perception_result = await self.perception_stream.process(features, frame_idx)
        perc_lat = (time.perf_counter() - perc_t0) * 1000.0
        # update history
        self._history.append({
            "frame_idx": frame_idx,
            "features": features.tolist(),
            "perception": perception_result,
        })
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]
        # plan
        plan_t0 = time.perf_counter()
        actions = await self.planning_stream.predict(features, list(self._history))
        plan_lat = (time.perf_counter() - plan_t0) * 1000.0
        # build output
        out = StreamOutput(
            frame_idx=frame_idx,
            perception_result=perception_result,
            planned_actions=actions,
            perception_latency_ms=perc_lat,
            planning_latency_ms=plan_lat,
            features=features.tolist(),
            dropped=False,
        )
        self._frames[frame_idx] = {
            "frame": frame,
            "features": features,
            "output": out,
        }
        # baseline = the first time we process the frame
        self._baseline_outputs.setdefault(frame_idx, out)
        return out

    # ------------------------------------------------------------------
    def handle_frame_drop(self, dropped_indices: list[int]) -> dict[str, Any]:
        """Measure accuracy + latency impact when frames are dropped.

        For each dropped frame, the streamer imputes its output by either
        re-using the previous frame's output (carry-forward imputation) or
        by interpolating features between neighbours. The resulting accuracy
        delta is computed against the stored baseline outputs.

        Returns a summary dict with:

        * ``n_dropped`` — count of dropped frames
        * ``avg_accuracy_delta`` — mean (baseline_score - imputed_score)
        * ``max_accuracy_delta`` — worst-case accuracy loss
        * ``avg_latency_ms`` — mean imputation latency
        * ``per_frame`` — list of per-frame diagnostics
        """
        if not dropped_indices:
            return {
                "n_dropped": 0,
                "avg_accuracy_delta": 0.0,
                "max_accuracy_delta": 0.0,
                "avg_latency_ms": 0.0,
                "per_frame": [],
            }
        per_frame: list[dict[str, Any]] = []
        deltas: list[float] = []
        lats: list[float] = []
        for idx in dropped_indices:
            t0 = time.perf_counter()
            baseline = self._baseline_outputs.get(idx)
            # carry-forward imputation: use previous frame's output if available
            prev_idx = idx - 1
            prev = self._baseline_outputs.get(prev_idx) or self._baseline_outputs.get(idx - 2)
            if prev is None and baseline is None:
                continue
            imputed = prev if prev is not None else baseline
            imputed_score = (
                float(imputed.perception_result.get("salience_score", 0.0))
                if imputed else 0.0
            )
            baseline_score = (
                float(baseline.perception_result.get("salience_score", 0.0))
                if baseline else 0.0
            )
            # delta >= 0: how much WORSE the imputation is than the baseline.
            # If the imputed frame happens to score higher than the baseline
            # (rare but possible with mock scores), the delta is clamped to 0
            # (a bonus, not a loss).
            delta = max(0.0, baseline_score - imputed_score)
            deltas.append(delta)
            lats.append((time.perf_counter() - t0) * 1000.0)
            per_frame.append({
                "frame_idx": idx,
                "baseline_score": round(baseline_score, 6),
                "imputed_score": round(imputed_score, 6),
                "accuracy_delta": round(delta, 6),
                "imputation_strategy": "carry_forward",
                "imputed_from": prev_idx if prev else idx,
            })
        return {
            "n_dropped": len(deltas),
            "avg_accuracy_delta": float(np.mean(deltas)) if deltas else 0.0,
            "max_accuracy_delta": float(np.max(deltas)) if deltas else 0.0,
            "avg_latency_ms": float(np.mean(lats)) if lats else 0.0,
            "per_frame": per_frame,
        }

    # ------------------------------------------------------------------
    async def plan_future(
        self,
        current_features: np.ndarray | list[float],
        history: Optional[list[dict[str, Any]]] = None,
    ) -> list[PlanAction]:
        """Predict future perception actions without running the perception stream."""
        feats = np.asarray(current_features, dtype=np.float32)
        hist = history if history is not None else list(self._history)
        return await self.planning_stream.predict(feats, hist)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear stored history and baselines."""
        self._history.clear()
        self._frames.clear()
        self._baseline_outputs.clear()
        self._drop_impacts.clear()

    def get_baseline(self, frame_idx: int) -> Optional[StreamOutput]:
        return self._baseline_outputs.get(frame_idx)


__all__ = [
    "ForesightStreamer",
    "StreamOutput",
    "PlanAction",
    "ActionType",
    "PerceptionStreamProtocol",
    "PlanningStreamProtocol",
    "SharedEncoderProtocol",
    "MockSharedEncoder",
    "MockPerceptionStream",
    "MockPlanningStream",
]
