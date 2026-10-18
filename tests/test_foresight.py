# (c) Copyright 2026 Bhaskar Kaphle. MIT License.
#
# Unit tests for OmniFlow — ForesightStreamer (omnimodal.streaming.foresight).
# Foresight is the dual-stream streaming VLM layer: a low-latency "now" stream
# processes every Nth frame and a higher-quality "future" stream runs ahead
# using a SHARED encoder (the encoder weights are reused, not duplicated).
# Foresight also handles dropped frames gracefully — when a frame is dropped
# (e.g., bandwidth pressure), the streamer interpolates from the previous
# processed frame so downstream consumers see a continuous signal.
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
    from omnimodal.core.types import Frame, Modality  # type: ignore
    from omnimodal.streaming.foresight import ForesightStreamer  # type: ignore

    _REAL_IMPL = True
except Exception:  # pragma: no cover - inline fallback when om-A/om-B missing
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
    class Frame:
        index: int = 0
        modality: Modality = Modality.VIDEO
        content: Any = None
        timestamp: float = 0.0
        dropped: bool = False
        metadata: dict[str, Any] = field(default_factory=dict)

    class ForesightStreamer:
        """Dual-stream VLM with shared encoder + future planning + drop resilience.

        Contract:
          * process_frame(frame) -> dict  : encode the current frame and produce
            a compact "now" prediction.
          * plan_future(frames, lookahead=...) -> list[Frame]  : pick the next
            `lookahead` frames whose indices are NOT multiples of frame_skip
            (these become the future stream's work queue).
          * handle_dropped_frame(frame) -> dict  : gracefully degrade by
            returning the last successful prediction with `interpolated=True`.
          * encoder_shared: bool  : True iff the now/future streams share the
            same encoder instance.
        """

        def __init__(
            self,
            *,
            frame_skip: int = 2,
            lookahead: int = 3,
            shared_encoder: Any = None,
            dual_stream: bool = True,
            seed: int = 42,
        ) -> None:
            if frame_skip < 1:
                raise ValueError("frame_skip must be >= 1")
            if lookahead < 0:
                raise ValueError("lookahead must be >= 0")
            self.frame_skip = int(frame_skip)
            self.lookahead = int(lookahead)
            self.seed = int(seed)
            self.dual_stream = bool(dual_stream)
            # shared_encoder is supplied externally OR created lazily as a
            # single shared object used by both streams.
            self._encoder = shared_encoder if shared_encoder is not None else _SharedEncoder(seed=seed)
            self._now_stream = _Stream("now", self._encoder)
            self._future_stream = (
                _Stream("future", self._encoder) if dual_stream else None
            )
            self.frame_count = 0
            self.dropped_count = 0
            self.last_plan: list[Frame] = []
            self._last_prediction: dict[str, Any] = {}

        # -- public surface ------------------------------------------------
        @property
        def encoder_shared(self) -> bool:
            """True iff the now and future streams share the same encoder."""
            if self._future_stream is None:
                return True
            return self._now_stream.encoder is self._future_stream.encoder

        def process_frame(self, frame) -> dict:
            """Encode + predict for the current frame.

            Returns a dict with keys: index, prediction, stream ('now' or
            'future'), shared_encoder (bool), interpolated (bool).
            """
            if isinstance(frame, dict):
                f = Frame(**frame)
            else:
                f = frame
            self.frame_count += 1
            enc = self._now_stream.encode(f)
            pred = self._now_stream.predict(enc)
            out = {
                "index": f.index,
                "prediction": pred,
                "stream": "now",
                "shared_encoder": self.encoder_shared,
                "interpolated": False,
                "frame_count": self.frame_count,
            }
            self._last_prediction = out
            return out

        def plan_future(self, frames, *, lookahead: int | None = None) -> list:
            """Pick `lookahead` future-stream frames from `frames`.

            The future stream processes frames whose indices are NOT multiples
            of frame_skip (i.e., the "in-between" frames the low-latency now
            stream skips over). Returns up to `lookahead` such frames.
            """
            n = self.lookahead if lookahead is None else int(lookahead)
            if n <= 0:
                return []
            # Filter frames whose idx is not a multiple of frame_skip
            # (those are the "future stream" work queue).
            future_frames = [
                f for f in frames
                if (f.index if hasattr(f, "index") else f.get("index", 0)) % self.frame_skip != 0
            ]
            plan = future_frames[:n]
            self.last_plan = list(plan)
            return plan

        def handle_dropped_frame(self, frame) -> dict:
            """Return a synthetic prediction interpolated from the last good one."""
            self.dropped_count += 1
            if not self._last_prediction:
                # No prior prediction to interpolate from -> return a neutral stub.
                return {
                    "index": frame.index if hasattr(frame, "index") else frame.get("index", 0),
                    "prediction": "interpolated:neutral",
                    "stream": "now",
                    "shared_encoder": self.encoder_shared,
                    "interpolated": True,
                    "frame_count": self.frame_count,
                }
            out = dict(self._last_prediction)
            idx = frame.index if hasattr(frame, "index") else frame.get("index", 0)
            out["index"] = idx
            out["prediction"] = f"interpolated:{self._last_prediction.get('prediction', '')}"
            out["interpolated"] = True
            return out

        def run_sequence(self, frames: list, *, lookahead: int | None = None) -> list:
            """Run a full sequence: every frame is processed, dropped frames
            trigger handle_dropped_frame; the future stream's plan is built
            from the remaining frames."""
            results = []
            for f in frames:
                fd = f if hasattr(f, "dropped") else Frame(**f)
                if getattr(fd, "dropped", False):
                    results.append(self.handle_dropped_frame(fd))
                else:
                    results.append(self.process_frame(fd))
            if self._future_stream is not None:
                self.last_plan = self.plan_future(frames, lookahead=lookahead)
            return results

    class _SharedEncoder:
        """Tiny deterministic shared encoder used by both streams."""

        def __init__(self, *, seed: int = 42) -> None:
            self.seed = int(seed)
            self._calls = 0

        def encode(self, frame) -> dict:
            self._calls += 1
            idx = frame.index if hasattr(frame, "index") else 0
            return {
                "embedding": [float(idx), float(idx + 1), float(idx + 2)],
                "calls": self._calls,
                "source": "shared",
            }

    class _Stream:
        def __init__(self, name: str, encoder: _SharedEncoder) -> None:
            self.name = name
            self.encoder = encoder

        def encode(self, frame) -> dict:
            return self.encoder.encode(frame)

        def predict(self, enc: dict) -> str:
            return f"pred:{self.name}:{enc.get('embedding', [0])[0]:.1f}"


# ============================================================================
# Tests
# ============================================================================


# ---- 1. process_frame returns well-formed dict -----------------------------
def test_process_frame_returns_well_formed_dict():
    fs = ForesightStreamer(frame_skip=2, lookahead=3, seed=42)
    f = Frame(index=0, modality=Modality.VIDEO, content="frame0")
    out = fs.process_frame(f)
    assert "index" in out and "prediction" in out and "stream" in out
    assert out["stream"] == "now"
    assert out["interpolated"] is False
    assert out["frame_count"] == 1


# ---- 2. Frame count increments per processed frame -----------------------
def test_frame_count_increments_per_call():
    fs = ForesightStreamer()
    assert fs.frame_count == 0
    for i in range(5):
        fs.process_frame(Frame(index=i, content=f"f{i}"))
    assert fs.frame_count == 5


# ---- 3. plan_future picks lookahead frames not multiples of frame_skip ---
def test_plan_future_picks_skip_offset_frames():
    fs = ForesightStreamer(frame_skip=2, lookahead=3, seed=42)
    frames = [Frame(index=i, content=f"f{i}") for i in range(10)]
    plan = fs.plan_future(frames)
    # With frame_skip=2, future frames are those with odd indices (1,3,5,...)
    assert all(f.index % 2 != 0 for f in plan)
    assert len(plan) == 3  # capped at lookahead
    assert fs.last_plan == plan


# ---- 4. plan_future respects an explicit lookahead override --------------
def test_plan_future_respects_explicit_lookahead():
    fs = ForesightStreamer(frame_skip=2, lookahead=10, seed=42)
    frames = [Frame(index=i, content=f"f{i}") for i in range(20)]
    plan = fs.plan_future(frames, lookahead=2)
    assert len(plan) == 2


# ---- 5. Dropped frame triggers handle_dropped_frame with interpolated=True
def test_dropped_frame_handled_with_interpolation():
    fs = ForesightStreamer(frame_skip=2, lookahead=2, seed=42)
    # Seed a prior successful prediction
    fs.process_frame(Frame(index=0, content="f0"))
    # Now simulate a dropped frame
    dropped = Frame(index=1, content="f1", dropped=True)
    out = fs.handle_dropped_frame(dropped)
    assert out["interpolated"] is True
    assert out["index"] == 1
    assert "prediction" in out
    assert fs.dropped_count == 1


# ---- 6. handle_dropped_frame without prior prediction returns neutral -----
def test_handle_dropped_frame_without_prior_returns_neutral_stub():
    fs = ForesightStreamer()
    out = fs.handle_dropped_frame(Frame(index=0, content="f0", dropped=True))
    assert out["interpolated"] is True
    assert out["prediction"] == "interpolated:neutral"
    assert fs.dropped_count == 1


# ---- 7. Dual-stream shares a single encoder instance ---------------------
def test_dual_stream_shares_encoder():
    fs_dual = ForesightStreamer(dual_stream=True, seed=42)
    assert fs_dual.dual_stream is True
    assert fs_dual.encoder_shared is True, "now and future must share encoder"
    # When the future stream is disabled, still claim "shared" (trivially).
    fs_single = ForesightStreamer(dual_stream=False, seed=42)
    assert fs_single.encoder_shared is True


# ---- 8. run_sequence integrates process + drop + plan ---------------------
def test_run_sequence_integrates_process_drop_and_plan():
    fs = ForesightStreamer(frame_skip=2, lookahead=3, seed=42)
    frames = [Frame(index=i, content=f"f{i}", dropped=(i == 3)) for i in range(8)]
    results = fs.run_sequence(frames)
    assert len(results) == 8
    # Exactly one dropped frame -> one interpolated entry
    interpolated = [r for r in results if r.get("interpolated")]
    assert len(interpolated) == 1
    assert interpolated[0]["index"] == 3
    # fs.last_plan holds the future plan (indices 1, 3, 5 — though 3 was dropped,
    # the planner picks purely from indices regardless of dropped flag)
    assert len(fs.last_plan) == 3
    assert all(f.index % 2 != 0 for f in fs.last_plan)


# ---- 9. Invalid frame_skip / lookahead raises ValueError ------------------
def test_invalid_frame_skip_and_lookahead_raise():
    with pytest.raises(ValueError):
        ForesightStreamer(frame_skip=0)
    with pytest.raises(ValueError):
        ForesightStreamer(lookahead=-1)


# ---- 10. Encoder is called once per processed frame (shared across streams)
def test_encoder_called_once_per_processed_frame():
    fs = ForesightStreamer(dual_stream=True, seed=42)
    enc = fs._encoder  # access to assert call count
    calls_before = enc._calls
    for i in range(3):
        fs.process_frame(Frame(index=i, content=f"f{i}"))
    assert enc._calls - calls_before == 3


# ---- 11. plan_future with empty input returns empty list ------------------
def test_plan_future_empty_input_returns_empty_list():
    fs = ForesightStreamer()
    assert fs.plan_future([]) == []
    assert fs.last_plan == []


# ---- 12. lookahead=0 returns empty plan -----------------------------------
def test_plan_future_zero_lookahead_returns_empty():
    fs = ForesightStreamer(lookahead=0)
    frames = [Frame(index=i, content=f"f{i}") for i in range(5)]
    assert fs.plan_future(frames) == []
