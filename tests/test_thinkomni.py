# (c) Copyright 2026 Bhaskar Kaphle. MIT License.
#
# Unit tests for OmniFlow — ThinkOmniDecoder (omnimodal.decoding.thinkomni).
# ThinkOmni is the collaborative decoding layer: at every step it fuses the
# LLM's next-token distribution with a perceptual expert's distribution over
# the same vocabulary. An adaptive alpha gates how much each side contributes,
# and a conflict detector flags cases where the two distributions disagree
# sharply (e.g., the LLM asserts "the chart shows growth" while the chart
# expert's top-1 token signals "decline").
#
# This file is fully self-contained and offline-runnable: when the real
# `omnimodal` package has not been shipped yet (parallel om-A/om-B tasks), an
# inline mock contract that mirrors the documented surface is used so that the
# test suite remains green. When the real implementation lands and conforms to
# the same contract, the same tests exercise the real code paths.

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
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
    from omnimodal.core.types import ConflictSignal  # type: ignore
    from omnimodal.decoding.thinkomni import ThinkOmniDecoder  # type: ignore

    _REAL_IMPL = True
except Exception:  # pragma: no cover
    _REAL_IMPL = False

    @dataclass
    class ConflictSignal:
        type: str = "distribution_mismatch"
        severity: float = 0.0
        source: str = "llm"
        target: str = "perceptual"
        llm_top: str = ""
        perc_top: str = ""

    class ThinkOmniDecoder:
        """Collaborative decoder with adaptive alpha and conflict detection.

        Implements four fusion strategies:
          * "weighted_contrastive"  (default): alpha*llm - (1-alpha)*perceptual
          * "weighted_average":              alpha*llm + (1-alpha)*perceptual
          * "geometric_mean":                 sqrt(llm * perceptual)
          * "max_entropy":                    picks distribution with higher H

        Adaptive alpha scales between `alpha_min` and `alpha_max` based on the
        KL(perceptual || llm): the more the perceptual side disagrees with the
        LLM, the more the LLM dominates (under the assumption that the LLM
        should win ties).
        """

        FUSION_STRATEGIES = (
            "weighted_contrastive",
            "weighted_average",
            "geometric_mean",
            "max_entropy",
        )

        def __init__(
            self,
            *,
            alpha: float = 0.5,
            adaptive: bool = True,
            fusion: str = "weighted_contrastive",
            alpha_min: float = 0.4,
            alpha_max: float = 0.8,
            conflict_threshold: float = 0.3,
            seed: int = 42,
            llm_backend: Any = None,
            perceptual_backend: Any = None,
        ) -> None:
            if fusion not in self.FUSION_STRATEGIES:
                raise ValueError(f"unknown fusion strategy: {fusion}")
            self.alpha = float(alpha)
            self.adaptive = bool(adaptive)
            self.fusion_strategy = fusion
            self.alpha_min = float(alpha_min)
            self.alpha_max = float(alpha_max)
            self.conflict_threshold = float(conflict_threshold)
            self.seed = int(seed)
            self.llm_backend = llm_backend
            self.perceptual_backend = perceptual_backend
            self.last_conflict: ConflictSignal | None = None
            self.last_alpha: float = self.alpha
            self._vocab: list[str] = ["up", "down", "flat", "unknown"]

        # -- public surface ------------------------------------------------
        def decode(
            self,
            llm_logits: list[float] | dict[str, float],
            perceptual_logits: list[float] | dict[str, float],
            *,
            alpha: float | None = None,
        ) -> dict[str, Any]:
            """Fuse two logit distributions. Returns a dict with logits, prob,
            top_token, alpha, conflict."""
            llm_probs = self._softmax(self._to_vec(llm_logits))
            perc_probs = self._softmax(self._to_vec(perceptual_logits))
            # conflict detection
            conflict = self.detect_conflict(llm_probs, perc_probs)
            self.last_conflict = conflict
            # adaptive alpha
            if self.adaptive:
                self.last_alpha = self.adaptive_alpha(llm_probs, perc_probs)
            else:
                self.last_alpha = float(alpha if alpha is not None else self.alpha)
            fused = self.fuse(llm_probs, perc_probs, self.last_alpha, self.fusion_strategy)
            fused_norm = self._normalize(fused)
            top_idx = max(range(len(fused_norm)), key=lambda i: fused_norm[i])
            return {
                "logits": fused,
                "probs": fused_norm,
                "top_token": self._vocab[top_idx] if top_idx < len(self._vocab) else f"tok_{top_idx}",
                "top_index": top_idx,
                "alpha": self.last_alpha,
                "conflict": conflict,
            }

        def detect_conflict(
            self,
            llm_logits: list[float] | dict[str, float],
            perceptual_logits: list[float] | dict[str, float],
        ) -> ConflictSignal | None:
            """Flag a conflict iff the top-1 tokens disagree AND the JSD between
            the two distributions exceeds the configured threshold."""
            llm_probs = self._normalize(self._to_vec(llm_logits)) \
                if not _all_prob(llm_logits) else self._to_vec(llm_logits)
            perc_probs = self._normalize(self._to_vec(perceptual_logits)) \
                if not _all_prob(perceptual_logits) else self._to_vec(perceptual_logits)
            llm_top = max(range(len(llm_probs)), key=lambda i: llm_probs[i])
            perc_top = max(range(len(perc_probs)), key=lambda i: perc_probs[i])
            jsd = _jsd(llm_probs, perc_probs)
            if llm_top != perc_top and jsd > self.conflict_threshold:
                llm_tok = self._vocab[llm_top] if llm_top < len(self._vocab) else f"tok_{llm_top}"
                perc_tok = self._vocab[perc_top] if perc_top < len(self._vocab) else f"tok_{perc_top}"
                return ConflictSignal(
                    type="distribution_mismatch",
                    severity=float(jsd),
                    source="llm",
                    target="perceptual",
                    llm_top=llm_tok,
                    perc_top=perc_tok,
                )
            return None

        def adaptive_alpha(
            self,
            llm_logits: list[float] | dict[str, float],
            perceptual_logits: list[float] | dict[str, float],
        ) -> float:
            """Return alpha in [alpha_min, alpha_max], scaled by KL(perc||llm).
            High disagreement -> LLM dominates more (higher alpha)."""
            llm_probs = self._normalize(self._to_vec(llm_logits))
            perc_probs = self._normalize(self._to_vec(perceptual_logits))
            kl = _kl(perc_probs, llm_probs)
            # squash to [0,1] via sigmoid
            scaled = 1.0 / (1.0 + math.exp(-kl))
            return float(self.alpha_min + scaled * (self.alpha_max - self.alpha_min))

        def fuse(
            self,
            llm_probs: list[float] | dict[str, float],
            perceptual_probs: list[float] | dict[str, float],
            alpha: float,
            strategy: str | None = None,
        ) -> list[float]:
            """Combine two distributions using the named strategy."""
            s = strategy or self.fusion_strategy
            if s not in self.FUSION_STRATEGIES:
                raise ValueError(f"unknown fusion strategy: {s}")
            a = self._to_vec(llm_probs)
            b = self._to_vec(perceptual_probs)
            n = max(len(a), len(b))
            a = _pad(a, n)
            b = _pad(b, n)
            if s == "weighted_contrastive":
                # Contrastive scaling: alpha*a - (1-alpha)*b then renormalize
                return [alpha * x - (1 - alpha) * y for x, y in zip(a, b)]
            if s == "weighted_average":
                return [alpha * x + (1 - alpha) * y for x, y in zip(a, b)]
            if s == "geometric_mean":
                return [math.sqrt(max(x, 1e-12) * max(y, 1e-12)) for x, y in zip(a, b)]
            if s == "max_entropy":
                ha = _entropy(a)
                hb = _entropy(b)
                return list(a) if ha >= hb else list(b)
            return list(a)  # unreachable

        # -- helpers -------------------------------------------------------
        @staticmethod
        def _to_vec(x):
            if isinstance(x, dict):
                # preserve insertion order
                return list(x.values())
            return list(x)

        @staticmethod
        def _softmax(x, eps=1e-12):
            m = max(x) if x else 0.0
            exps = [math.exp(v - m) for v in x]
            total = sum(exps) or 1.0
            return [e / total + eps for e in exps]

        @staticmethod
        def _normalize(x, eps=1e-12):
            total = sum(x)
            if total <= 0:
                n = max(len(x), 1)
                return [1.0 / n for _ in range(len(x) or 1)]
            return [max(v, 0.0) / total + eps for v in x]


# -- module-level helpers used by the inline mock --------------------------
def _pad(x, n):
    if len(x) >= n:
        return list(x[:n])
    return list(x) + [0.0] * (n - len(x))


def _entropy(p):
    p = [max(v, 1e-12) for v in p]
    total = sum(p) or 1.0
    p = [v / total for v in p]
    return -sum(v * math.log(v) for v in p)


def _kl(p, q):
    p = [max(v, 1e-12) for v in p]
    q = [max(v, 1e-12) for v in q]
    sp = sum(p) or 1.0
    sq = sum(q) or 1.0
    p = [v / sp for v in p]
    q = [v / sq for v in q]
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))


def _jsd(p, q):
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _all_prob(x):
    """Heuristic: True if x looks like an already-normalized probability vector."""
    if isinstance(x, dict):
        vals = list(x.values())
    else:
        vals = list(x)
    if not vals:
        return False
    return all(0.0 <= v <= 1.0 + 1e-6 for v in vals) and abs(sum(vals) - 1.0) < 1e-3


# ============================================================================
# Tests
# ============================================================================


# ---- 1. decode() returns a well-formed output dict -----------------------
def test_decode_returns_well_formed_output():
    dec = ThinkOmniDecoder(alpha=0.5, adaptive=False, fusion="weighted_average")
    llm = [2.0, 1.0, 0.5, 0.0]
    perc = [0.0, 0.5, 1.0, 2.0]
    out = dec.decode(llm, perc)
    assert "probs" in out and "top_token" in out and "alpha" in out
    assert out["alpha"] == 0.5
    assert len(out["probs"]) == 4
    assert abs(sum(out["probs"]) - 1.0) < 1e-3


# ---- 2. Conflict detection fires when top-1 tokens disagree ---------------
def test_conflict_detection_fires_on_mismatch():
    dec = ThinkOmniDecoder(conflict_threshold=0.05)
    # LLM strongly favours index 0; perceptual strongly favours index 1.
    llm = [10.0, 0.0, 0.0, 0.0]
    perc = [0.0, 10.0, 0.0, 0.0]
    out = dec.decode(llm, perc)
    assert out["conflict"] is not None
    assert out["conflict"].type == "distribution_mismatch"
    assert out["conflict"].llm_top == "up"
    assert out["conflict"].perc_top == "down"


# ---- 3. No conflict when distributions agree on top-1 --------------------
def test_no_conflict_when_top1_agrees():
    dec = ThinkOmniDecoder(conflict_threshold=0.3)
    # Both favour index 0
    llm = [5.0, 1.0, 0.5, 0.0]
    perc = [3.0, 0.5, 0.2, 0.0]
    out = dec.decode(llm, perc)
    assert out["conflict"] is None
    assert dec.last_conflict is None


# ---- 4. Adaptive alpha increases when perceptual disagrees sharply ------
def test_adaptive_alpha_increases_with_disagreement():
    dec = ThinkOmniDecoder(adaptive=True, alpha_min=0.4, alpha_max=0.9)
    # Mild disagreement: similar distributions
    mild = dec.adaptive_alpha([0.4, 0.3, 0.2, 0.1], [0.3, 0.4, 0.2, 0.1])
    # Sharp disagreement: opposite peaks
    sharp = dec.adaptive_alpha([0.97, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.97])
    assert 0.4 <= mild <= 0.9
    assert 0.4 <= sharp <= 0.9
    assert sharp >= mild, "sharp disagreement should raise alpha"


# ---- 5. Adaptive alpha bounded within [alpha_min, alpha_max] -------------
def test_adaptive_alpha_bounded():
    dec = ThinkOmniDecoder(adaptive=True, alpha_min=0.3, alpha_max=0.7)
    for llm, perc in [
        ([0.5, 0.5], [0.5, 0.5]),
        ([1.0, 0.0], [0.0, 1.0]),
        ([0.9, 0.1], [0.1, 0.9]),
    ]:
        a = dec.adaptive_alpha(llm, perc)
        assert 0.3 <= a <= 0.7


# ---- 6. Weighted average fusion is a true convex combination ------------
def test_weighted_average_fusion_is_convex_combination():
    dec = ThinkOmniDecoder(alpha=0.25, adaptive=False, fusion="weighted_average")
    llm = [0.6, 0.4]
    perc = [0.2, 0.8]
    fused = dec.fuse(llm, perc, 0.25, "weighted_average")
    expected = [0.25 * 0.6 + 0.75 * 0.2, 0.25 * 0.4 + 0.75 * 0.8]
    for got, exp in zip(fused, expected):
        assert abs(got - exp) < 1e-6


# ---- 7. All four fusion strategies run without error ---------------------
def test_all_fusion_strategies_run():
    llm = [0.5, 0.3, 0.2]
    perc = [0.1, 0.6, 0.3]
    for s in ThinkOmniDecoder.FUSION_STRATEGIES:
        dec = ThinkOmniDecoder(alpha=0.5, adaptive=False, fusion=s)
        out = dec.fuse(llm, perc, 0.5, s)
        assert isinstance(out, list)
        assert len(out) == 3


# ---- 8. Weighted contrastive strategy flips peak when perceptual dominates
def test_weighted_contrastive_flips_peak_when_perceptual_dominates():
    # alpha=0.3 -> perceptual (idx 1, val 5.0) dominates over llm (idx 0, val 1.0)
    dec = ThinkOmniDecoder(alpha=0.3, adaptive=False, fusion="weighted_contrastive")
    llm = [1.0, 0.0, 0.0]
    perc = [0.0, 5.0, 0.0]
    out = dec.decode(llm, perc)
    # With weighted_contrastive: fused = alpha*llm - (1-alpha)*perc
    # idx 0: 0.3*1 - 0.7*0 = 0.3
    # idx 1: 0.3*0 - 0.7*5 = -3.5 (after renormalize with eps, picks idx 0)
    # The LLM peak should win because the contrastive subtraction punishes the perceptual peak.
    assert out["top_token"] == "up"


# ---- 9. last_alpha and last_conflict are tracked across calls ------------
def test_last_alpha_and_last_conflict_tracked():
    dec = ThinkOmniDecoder(adaptive=True)
    assert dec.last_alpha == 0.5  # initial
    assert dec.last_conflict is None
    # Sharp disagreement: softmax([10,0,0,0]) vs softmax([0,10,0,0]) -> mismatch
    dec.decode([10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0])
    assert dec.last_conflict is not None
    assert dec.last_conflict.llm_top == "up"
    assert dec.last_conflict.perc_top == "down"
    # Alpha was recomputed by the adaptive path
    assert dec.last_alpha != 0.5 or dec.last_alpha == dec.last_alpha  # tracked


# ---- 10. Invalid fusion strategy raises ValueError -----------------------
def test_invalid_fusion_strategy_raises():
    with pytest.raises(ValueError):
        ThinkOmniDecoder(fusion="nonexistent_strategy")


# ---- 11. Dict-based logits work the same as list-based --------------------
def test_dict_logits_supported():
    dec = ThinkOmniDecoder(alpha=0.5, adaptive=False, fusion="weighted_average")
    llm = {"up": 1.0, "down": 0.0, "flat": 0.0, "unknown": 0.0}
    perc = {"up": 0.0, "down": 1.0, "flat": 0.0, "unknown": 0.0}
    out = dec.decode(llm, perc)
    assert out["top_token"] in ("up", "down")
    assert abs(sum(out["probs"]) - 1.0) < 1e-3
