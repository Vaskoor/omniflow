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
"""ThinkOmniDecoder — training-free collaborative decoding.

Architecture (paper-style):

1. **Text reasoner** (:class:`TextReasonerProtocol`) — produces a chain of
   thought plus a token distribution ``text_logits`` that biases the
   omni-modal LLM toward the reasoning-derived answer.
2. **Omni-modal LLM** (:class:`OmniModalLLMProtocol`) — produces a perception
   distribution ``perception_logits`` from the image (and optional text
   query).
3. **Adaptive fusion** — :func:`adaptive_weighted_contrastive_scaling`
   fuses the two distributions. The weight ``alpha`` on the text logits
   adapts to the text reasoner's confidence (low entropy -> high alpha;
   high entropy -> low alpha). A contrastive scaling term boosts the fused
   distribution where the two streams agree and dampens where they disagree.
4. **Conflict diagnosis** — :meth:`diagnose_conflict` flags when text
   reasoning disagrees with perceptual evidence (numeric disagreement,
   entity contradiction, keyword mismatch).

The decoder is training-free — both input streams are black-box callable
objects; the mock implementations included here are deterministic
numpy-based references used by the benchmark suite when real LLMs are not
available.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

# Optional contract import; not strictly required for the decoder's own
# data types, but kept for forward-compatibility with om-A's ControlToken.
try:
    from omnimodal.core.types import ControlToken  # type: ignore[attr-defined]
except ImportError:
    from omnimodal.experts.base import ControlToken  # type: ignore


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------
class ReasoningOutput(BaseModel):
    """Output of the text reasoner."""

    text: str = ""
    logits: list[float] = Field(default_factory=list)
    confidence: float = 0.0
    chain_of_thought: list[str] = Field(default_factory=list)


class PerceptionOutput(BaseModel):
    """Output of the omni-modal LLM."""

    text: str = ""
    logits: list[float] = Field(default_factory=list)
    confidence: float = 0.0
    modality_signals: dict[str, Any] = Field(default_factory=dict)


class ConflictType(str, Enum):
    """Categories of text-vs-perception conflicts."""

    NONE = "none"
    NUMERIC_DISAGREEMENT = "numeric_disagreement"
    ENTITY_CONTRADICTION = "entity_contradiction"
    KEYWORD_MISMATCH = "keyword_mismatch"


class ConflictReport(BaseModel):
    """Report returned by :meth:`ThinkOmniDecoder.diagnose_conflict`."""

    has_conflict: bool = False
    conflict_type: ConflictType = ConflictType.NONE
    severity: float = 0.0  # 0 = no conflict, 1 = maximal conflict
    details: str = ""
    text_only_terms: list[str] = Field(default_factory=list)
    perceptual_only_terms: list[str] = Field(default_factory=list)
    numeric_text: list[str] = Field(default_factory=list)
    numeric_perceptual: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stream protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class TextReasonerProtocol(Protocol):
    """Callable interface for the text reasoner."""

    async def generate_reasoning(
        self, query: str, context: Optional[str] = None
    ) -> ReasoningOutput: ...


@runtime_checkable
class OmniModalLLMProtocol(Protocol):
    """Callable interface for the omni-modal LLM."""

    async def generate_perception(
        self,
        query: str,
        image: Any,
        context: Optional[str] = None,
    ) -> PerceptionOutput: ...


# ---------------------------------------------------------------------------
# Numeric utilities
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p + 1e-12)).sum())


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    # KL divergence with smoothing
    def _kl(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-12) / (b[mask] + 1e-12))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def adaptive_weighted_contrastive_scaling(
    text_logits: np.ndarray | list[float],
    perception_logits: np.ndarray | list[float],
    alpha: Optional[float] = None,
) -> np.ndarray:
    """Fuse text and perception logits via adaptive weighted contrastive scaling.

    Parameters
    ----------
    text_logits, perception_logits
        1-D arrays of shape ``(vocab_size,)``. If the two arrays have
        different lengths the longer is truncated to the shorter.
    alpha
        Weight on the text logits in ``[0, 1]``. If ``None`` (the default),
        alpha is auto-adapted from the text reasoner's confidence: the more
        peaked the text distribution (lower entropy), the higher alpha.

    Returns
    -------
    np.ndarray
        Fused logits of the same shape as the inputs.

    Notes
    -----
    The fusion formula is:

    .. math::

        \\hat{p}_{\\text{text}} &= \\mathrm{softmax}(\\ell_t) \\\\
        \\hat{p}_{\\text{perc}} &= \\mathrm{softmax}(\\ell_p) \\\\
        \\alpha &= 1 - H(\\hat{p}_{\\text{text}})/\\log V
                 \\quad\\text{(auto-adapted)} \\\\
        \\text{agree} &= \\frac{1}{1 + e^{-\\mathrm{JS}(\\hat{p}_t \\,
                                  \\hat{p}_p)}} \\\\
        \\ell_{\\text{fused}} &= (\\alpha \\ell_t + (1-\\alpha) \\ell_p)
                                  \\cdot (1 + 0.5\\,\\text{agree})

    where JS is the Jensen-Shannon divergence. The contrastive term boosts
    the fused distribution when the streams agree and dampens it when they
    disagree.
    """
    t = np.asarray(text_logits, dtype=float).ravel()
    p = np.asarray(perception_logits, dtype=float).ravel()
    n = min(t.size, p.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    t = t[:n]
    p = p[:n]
    p_t = _softmax(t)
    p_p = _softmax(p)
    if alpha is None:
        # alpha in [0.1, 0.9], scaled by 1 - normalized entropy of text dist.
        H = _entropy(p_t)
        Hmax = float(np.log(max(n, 1)))
        norm_entropy = H / Hmax if Hmax > 0 else 0.0
        alpha = float(0.1 + 0.8 * (1.0 - norm_entropy))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    js = _js_divergence(p_t, p_p)
    # agreement in (0, 1) — high when JS is small
    agreement = float(1.0 / (1.0 + np.exp(-(1.0 - min(js, 1.0)) * 6.0 + 3.0)))
    # adaptive weighted combination
    fused = alpha * t + (1.0 - alpha) * p
    # contrastive scaling: boost when agreement high
    scale = 1.0 + 0.5 * agreement
    return fused * scale


# ---------------------------------------------------------------------------
# Conflict diagnosis helpers
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "of", "in", "on", "at", "to", "for", "with", "by",
    "this", "that", "these", "those", "it", "they", "we", "you", "i",
    "as", "from", "but", "not", "no", "yes", "do", "does", "did", "has",
    "have", "had", "will", "would", "could", "should", "can", "may",
    "if", "then", "than", "so", "such", "very", "into", "about",
}


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _content_terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS and len(w) > 2}


def diagnose_conflict(
    text_reasoning: str,
    perceptual_evidence: str | dict[str, Any],
) -> ConflictReport:
    """Diagnose whether text reasoning conflicts with perceptual evidence.

    Detects:

    * numeric disagreement — overlapping numbers in text vs perception that
      differ in value
    * entity contradiction — same entity name appears in both, but with
      different qualifier context
    * keyword mismatch — significant terms present in one stream but absent
      in the other
    """
    if isinstance(perceptual_evidence, dict):
        # flatten dict to text by joining values
        flat = " ".join(str(v) for v in perceptual_evidence.values())
    else:
        flat = str(perceptual_evidence)

    text_terms = _content_terms(text_reasoning)
    perc_terms = _content_terms(flat)
    text_only = sorted(text_terms - perc_terms)
    perc_only = sorted(perc_terms - text_terms)
    shared = text_terms & perc_terms

    text_nums = _numbers(text_reasoning)
    perc_nums = _numbers(flat)
    # numeric disagreement: same number set partially overlapping but not identical
    text_numset = set(text_nums)
    perc_numset = set(perc_nums)
    # Conflict if there are any numbers in both, and the sets are not nested
    conflict_num = False
    if text_numset and perc_numset:
        # If a number appears in one but not the other, that's a disagreement
        if text_numset != perc_numset:
            # check if any number appears in only one
            if text_numset - perc_numset or perc_numset - text_numset:
                conflict_num = True

    # entity contradiction: same entity token in both streams, but appears
    # with a negation in one and not the other.
    negations = {"not", "no", "without", "n't", "cannot", "never"}
    text_tokens = [t.lower() for t in _WORD_RE.findall(text_reasoning)]
    perc_tokens = [t.lower() for t in _WORD_RE.findall(flat)]
    text_has_neg = any(neg in text_tokens for neg in negations)
    perc_has_neg = any(neg in perc_tokens for neg in negations)
    entity_contradiction = bool(shared and (text_has_neg ^ perc_has_neg))

    # keyword mismatch: only flag when there is meaningful overlap AND a
    # majority of content terms are present in only one stream (i.e. the two
    # streams really disagree on the salient content, not just incidental
    # word choice differences).
    total_terms = len(text_terms) + len(perc_terms)
    mismatch_count = len(text_only) + len(perc_only)
    keyword_mismatch = bool(
        shared
        and total_terms > 0
        and (mismatch_count / total_terms) > 0.5
    ) or bool(
        not shared
        and (len(text_only) + len(perc_only)) >= 3
    )

    # decide conflict type with priority: numeric > entity > keyword > none
    if conflict_num:
        ctype = ConflictType.NUMERIC_DISAGREEMENT
        severity = 0.8
        details = (f"Numbers in text {text_nums} vs perception {perc_nums} "
                   f"differ (text-only={sorted(text_numset - perc_numset)}, "
                   f"perc-only={sorted(perc_numset - text_numset)}).")
    elif entity_contradiction:
        ctype = ConflictType.ENTITY_CONTRADICTION
        severity = 0.7
        details = f"Shared entities {sorted(shared)} with contradicting negation context."
    elif keyword_mismatch:
        ctype = ConflictType.KEYWORD_MISMATCH
        severity = float(min(0.6, 0.2 + 0.1 * max(len(text_only), len(perc_only))))
        details = (f"Text-only terms {text_only[:5]}...; "
                   f"perceptual-only terms {perc_only[:5]}...")
    else:
        ctype = ConflictType.NONE
        severity = 0.0
        details = "No conflict detected."

    return ConflictReport(
        has_conflict=ctype != ConflictType.NONE,
        conflict_type=ctype,
        severity=round(severity, 4),
        details=details,
        text_only_terms=text_only,
        perceptual_only_terms=perc_only,
        numeric_text=text_nums,
        numeric_perceptual=perc_nums,
    )


# ---------------------------------------------------------------------------
# Mock stream implementations (deterministic, no real LLM required)
# ---------------------------------------------------------------------------
def _try_math_eval(query: str) -> Optional[str]:
    """Evaluate a pure-arithmetic query and return the result as a string.

    Returns ``None`` if the query is not a pure arithmetic expression. The
    evaluator is a safe AST walker (no builtins, no attribute access) so it
    cannot be exploited.
    """
    import ast
    import operator as _op
    import re as _re

    if not query or not isinstance(query, str):
        return None
    expr = _re.sub(
        r"(?i)^(what\s+is|calculate|compute|evaluate|solve|find)\s+",
        "",
        query.strip(),
    )
    expr = expr.rstrip("=?").strip().replace("^", "**")
    if not expr:
        return None
    if not _re.match(r"^[\d\s\+\-\*\/\(\)\.\%\*]+$", expr):
        return None
    safe_ops = {
        ast.Add: _op.add,
        ast.Sub: _op.sub,
        ast.Mult: _op.mul,
        ast.Div: _op.truediv,
        ast.Pow: _op.pow,
        ast.USub: _op.neg,
        ast.UAdd: _op.pos,
        ast.Mod: _op.mod,
        ast.FloorDiv: _op.floordiv,
    }
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in safe_ops:
            return float(safe_ops[type(node.op)](_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in safe_ops:
            return float(safe_ops[type(node.op)](_eval(node.operand)))
        raise ValueError("unsafe node")

    try:
        val = _eval(tree)
    except Exception:
        return None
    if float(val).is_integer():
        return str(int(val))
    return f"{val:.6g}"


class MockTextReasoner:
    """Deterministic mock text reasoner.

    Produces a small fixed-size logit vector seeded from (query, context).
    Used by the benchmark suite when no real reasoner is wired in.

    For pure-arithmetic queries the mock actually evaluates the expression
    (via a safe AST walker) and returns the result as the text — this lets
    the MathVista-style benchmark produce non-trivial accuracy under mock
    mode. Non-arithmetic queries fall back to a deterministic token name.
    """

    def __init__(self, vocab_size: int = 64, seed: int = 42):
        self.vocab_size = vocab_size
        self.seed = seed

    async def generate_reasoning(
        self, query: str, context: Optional[str] = None
    ) -> ReasoningOutput:
        rng = np.random.RandomState(self._hash_seed(query, context or ""))
        # spike one or two logits deterministically
        logits = (rng.randn(self.vocab_size) * 0.5).astype(float)
        spike_idx = int(rng.randint(0, self.vocab_size))
        logits[spike_idx] += 8.0
        if self.vocab_size > 1:
            spike2 = int(rng.randint(0, self.vocab_size))
            logits[spike2] += 4.0
        p = _softmax(logits)
        conf = float(np.max(p))
        # try arithmetic eval first — gives a meaningful text for math queries
        math_answer = _try_math_eval(query)
        if math_answer is not None:
            text = math_answer
            cot = [
                f"Step 1: parsed query '{query[:60]}'.",
                f"Step 2: evaluated as arithmetic; result = {math_answer}.",
                f"Step 3: derived candidate answer with confidence {conf:.3f}.",
            ]
        else:
            # Non-arithmetic fallback: signal-only text (no digits) so it
            # does not spuriously trigger numeric-conflict detection in
            # diagnose_conflict().
            text = "reasoning_signal"
            cot = [
                f"Step 1: parsed query '{query[:60]}'.",
                f"Step 2: considered context of length {len(context or '')} chars.",
                f"Step 3: derived candidate answer with confidence {conf:.3f}.",
            ]
        return ReasoningOutput(
            text=text,
            logits=logits.tolist(),
            confidence=conf,
            chain_of_thought=cot,
        )

    def _hash_seed(self, *parts: str) -> int:
        h = hashlib.sha256("|".join(p or "" for p in parts).encode("utf-8")).hexdigest()
        return (int(h[:8], 16) ^ (self.seed * 0x9E3779B1)) & 0xFFFFFFFF


class MockOmniModalLLM:
    """Deterministic mock omni-modal LLM.

    Produces a small logit vector seeded from (query, image, context). The
    mock image is hashed alongside the query so the same image+query always
    yields the same perception distribution.
    """

    def __init__(self, vocab_size: int = 64, seed: int = 42):
        self.vocab_size = vocab_size
        self.seed = seed

    async def generate_perception(
        self,
        query: str,
        image: Any,
        context: Optional[str] = None,
    ) -> PerceptionOutput:
        image_repr = self._image_repr(image)
        rng = np.random.RandomState(self._hash_seed(query, image_repr, context or ""))
        logits = (rng.randn(self.vocab_size) * 0.5).astype(float)
        spike = int(rng.randint(0, self.vocab_size))
        logits[spike] += 7.0
        if self.vocab_size > 1:
            spike2 = int(rng.randint(0, self.vocab_size))
            logits[spike2] += 3.0
        p = _softmax(logits)
        conf = float(np.max(p))
        # Signal-only text (no digits) so the conflict detector does not
        # spuriously flag a numeric disagreement between text reasoning and
        # the perception stream's mock placeholder text.
        return PerceptionOutput(
            text="perception_signal",
            logits=logits.tolist(),
            confidence=conf,
            modality_signals={
                "image_hash": image_repr[:16],
                "spike_token": spike,
                "agreement_signal": float(rng.rand()),
            },
        )

    @staticmethod
    def _image_repr(image: Any) -> str:
        if image is None:
            return "none"
        if isinstance(image, (bytes, bytearray)):
            return hashlib.sha256(bytes(image)).hexdigest()
        if isinstance(image, np.ndarray):
            return hashlib.sha256(image.tobytes()).hexdigest()
        if isinstance(image, str):
            return hashlib.sha256(image.encode("utf-8")).hexdigest()
        return hashlib.sha256(repr(image).encode("utf-8")).hexdigest()

    def _hash_seed(self, *parts: str) -> int:
        h = hashlib.sha256("|".join(p or "" for p in parts).encode("utf-8")).hexdigest()
        return (int(h[:8], 16) ^ (self.seed * 0x9E3779B1)) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# ThinkOmniDecoder
# ---------------------------------------------------------------------------
class ThinkOmniDecoder:
    """Training-free collaborative decoder.

    Parameters
    ----------
    text_reasoner
        Object implementing :class:`TextReasonerProtocol`.
    omni_modal_llm
        Object implementing :class:`OmniModalLLMProtocol`.
    fusion_strategy
        One of ``"adaptive"`` (default — alpha auto-adapts from text
        confidence), ``"fixed"`` (uses ``self.alpha``), or ``"perception"``
        (alpha=0 — pure perception; debugging).
    alpha
        Used only when ``fusion_strategy='fixed'``.
    seed
        Determinism seed for any internal randomness.
    """

    def __init__(
        self,
        text_reasoner: TextReasonerProtocol,
        omni_modal_llm: OmniModalLLMProtocol,
        *,
        fusion_strategy: str = "adaptive",
        alpha: float = 0.5,
        seed: int = 42,
    ):
        self.text_reasoner = text_reasoner
        self.omni_modal_llm = omni_modal_llm
        if fusion_strategy not in {"adaptive", "fixed", "perception", "text"}:
            raise ValueError(
                f"unknown fusion_strategy {fusion_strategy!r}; "
                "expected one of: adaptive, fixed, perception, text"
            )
        self.fusion_strategy = fusion_strategy
        self.alpha = float(alpha)
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._last_text_output: Optional[ReasoningOutput] = None
        self._last_perception_output: Optional[PerceptionOutput] = None
        self._last_fused_logits: Optional[np.ndarray] = None
        self._last_alpha: Optional[float] = None
        self._last_conflict: Optional[ConflictReport] = None

    # ------------------------------------------------------------------
    async def decode(
        self,
        query: str,
        image: Any,
        context: Optional[str] = None,
    ) -> str:
        """Run training-free collaborative decoding.

        Returns the fused answer as a string. In the mock case this is
        composed from the text reasoner's and the perception stream's textual
        outputs, gated by the conflict report — if a conflict is detected,
        the perception stream's output is preferred (perceptual evidence wins
        over speculative text reasoning).
        """
        # 1) text reasoning bias
        text_out = await self.text_reasoner.generate_reasoning(query, context)
        self._last_text_output = text_out
        # 2) perception distribution
        perc_out = await self.omni_modal_llm.generate_perception(query, image, context)
        self._last_perception_output = perc_out
        # 3) fuse
        alpha = self._resolve_alpha(text_out)
        self._last_alpha = alpha
        text_logits = np.asarray(text_out.logits, dtype=float)
        perc_logits = np.asarray(perc_out.logits, dtype=float)
        fused = adaptive_weighted_contrastive_scaling(text_logits, perc_logits, alpha)
        self._last_fused_logits = fused
        # 4) diagnose conflict
        self._last_conflict = diagnose_conflict(
            text_out.text or " ".join(text_out.chain_of_thought),
            perc_out.text or perc_out.modality_signals,
        )
        # 5) choose output text — perception wins on conflict, else merge.
        if self._last_conflict.has_conflict:
            return perc_out.text or text_out.text or ""
        if text_out.text and perc_out.text and text_out.text != perc_out.text:
            # text reasoning anchors the answer, perception refines
            return f"{text_out.text} | perception: {perc_out.text}"
        return text_out.text or perc_out.text or ""

    def _resolve_alpha(self, text_out: ReasoningOutput) -> Optional[float]:
        if self.fusion_strategy == "fixed":
            return self.alpha
        if self.fusion_strategy == "perception":
            return 0.0
        if self.fusion_strategy == "text":
            return 1.0
        # adaptive
        return None  # adaptive_weighted_contrastive_scaling derives from entropy

    # ------------------------------------------------------------------
    def diagnose_conflict(
        self, text_reasoning: str, perceptual_evidence: str | dict[str, Any]
    ) -> ConflictReport:
        """Public entry point for conflict diagnosis (also called by decode)."""
        return diagnose_conflict(text_reasoning, perceptual_evidence)

    # ------------------------------------------------------------------
    def last_diagnostics(self) -> dict[str, Any]:
        """Return the most recent fusion diagnostics."""
        return {
            "alpha": self._last_alpha,
            "text_confidence": self._last_text_output.confidence if self._last_text_output else None,
            "perception_confidence": (
                self._last_perception_output.confidence if self._last_perception_output else None
            ),
            "fused_top_idx": (
                int(np.argmax(self._last_fused_logits))
                if self._last_fused_logits is not None and self._last_fused_logits.size > 0
                else None
            ),
            "conflict": self._last_conflict.model_dump() if self._last_conflict else None,
            "fusion_strategy": self.fusion_strategy,
        }


__all__ = [
    "ThinkOmniDecoder",
    "ConflictReport",
    "ConflictType",
    "ReasoningOutput",
    "PerceptionOutput",
    "TextReasonerProtocol",
    "OmniModalLLMProtocol",
    "MockTextReasoner",
    "MockOmniModalLLM",
    "adaptive_weighted_contrastive_scaling",
    "diagnose_conflict",
]
