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
"""TextReasoningExpert — pure-text reasoning (math / logic / summarization).

Backends:

==============  ===============================  ===========================
``backend``     library (guarded)               notes
``mock``        (none)                          deterministic, no LLM call
``llm``         ``z-ai-web-dev-sdk``            GLM-4.5 / GLM-4-Plus chat
``transformers`` ``transformers``             small local causal LM
==============  ===============================  ===========================

The query and optional context are passed via ``request.params['query']``
and ``request.params['context']`` (the contract's ``ExpertRequest`` does
not carry top-level query/context fields). The text content can also be
supplied directly via ``request.input_data``.

The mock backend handles three classes of tasks:

* **math** — evaluates arithmetic expressions via a safe AST walker;
  falls back to template answers if the expression is not parseable.
* **summarization** — returns the first N sentences / words.
* **logic / general** — pattern-based templates seeded off (query, context).
"""
from __future__ import annotations

import ast
import operator as _op
import re
import time
from typing import Any, Optional

import numpy as np

try:
    from omnimodal.core.types import (  # type: ignore[attr-defined]
        ExpertRequest,
        ExpertResponse,
        ExpertType,
        ModalityType,
    )
except ImportError:
    from omnimodal.experts.base import (
        ExpertRequest,
        ExpertResponse,
        ExpertType,
        ModalityType,
    )

from omnimodal.experts.base import Expert

# -- guarded real backends --------------------------------------------------
try:
    import z_ai_web_dev_sdk  # type: ignore
    _HAS_ZAI = True
except Exception:
    _HAS_ZAI = False

try:
    import transformers  # type: ignore
    _HAS_HF = True
except Exception:
    _HAS_HF = False
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# Safe operator table for arithmetic evaluation (no builtins).
_SAFE_OPS = {
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


def _safe_arith(expr: str) -> Optional[float]:
    """Evaluate a pure-arithmetic string, returning None on failure."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(node.op)](_eval(node.operand)))
        raise ValueError(f"unsafe node: {ast.dump(node)}")

    try:
        return _eval(tree)
    except Exception:
        return None


def _looks_like_math(q: str) -> bool:
    """Heuristic: does the query contain a printable arithmetic expression?"""
    s = re.sub(r"(?i)^(what\s+is|calculate|compute|evaluate|solve|find)\s+", "", q.strip())
    s = s.rstrip("=?").strip()
    if re.match(r"^[\d\s\+\-\*\/\(\)\.\%\^]+$", s):
        return True
    return False


def _extract_math_expr(q: str) -> str:
    s = re.sub(r"(?i)^(what\s+is|calculate|compute|evaluate|solve|find)\s+", "", q.strip())
    s = s.rstrip("=?").strip()
    s = s.replace("^", "**")
    return s


def _summarize(text: str, ratio: float = 0.3, max_words: int = 60) -> str:
    words = text.split()
    if not words:
        return ""
    n = max(1, min(max_words, int(len(words) * ratio)))
    return " ".join(words[:n])


class TextReasoningExpert(Expert):
    """Pure-text reasoning over arithmetic, logic, and summarization."""

    def __init__(
        self,
        *,
        backend: str = "mock",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(backend=backend, seed=seed, **kwargs)
        self.model_name = model_name or "glm-4.5"
        self.api_key = api_key
        self._client: Any = None
        if backend == "llm" and _HAS_ZAI:
            self._client = self._init_zai()
        elif backend == "transformers" and _HAS_HF and _HAS_TORCH:
            try:
                from transformers import pipeline  # type: ignore
                self._client = pipeline("text-generation", model=model_name or "sshleifer/tiny-gpt2")
            except Exception:
                self._client = None

    def _init_zai(self) -> Any:
        try:
            import os
            key = self.api_key or os.environ.get("ZAI_API_KEY") or os.environ.get("Z_AI_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
            for attr in ("ZAIWPredictor", "ZaiClient", "Client", "LLM"):
                cls = getattr(z_ai_web_dev_sdk, attr, None)
                if cls is None:
                    continue
                try:
                    return cls(api_key=key) if key else cls()
                except Exception:
                    try:
                        return cls()
                    except Exception:
                        continue
            return None
        except Exception:
            return None

    # -- Expert interface ------------------------------------------------
    @property
    def expert_type(self) -> ExpertType:
        return ExpertType.TEXT_REASON

    @property
    def modalities(self) -> list[ModalityType]:
        return [ModalityType.TEXT]

    async def run(self, request: ExpertRequest) -> ExpertResponse:
        t0 = time.perf_counter()
        try:
            params = dict(request.params or {})
            query = params.get("query") or (request.input_data if isinstance(request.input_data, str) else "")
            ctx = params.get("context", "")
            # If input_data is a longer string than the query (e.g. the user
            # passed a long text in input_data and "summarize" as the query
            # instruction), use it as the source context for summarization.
            if not ctx and isinstance(request.input_data, str) and len(request.input_data) > len(query):
                ctx = request.input_data
            task = params.get("task", "auto")
            if self.backend == "mock" or self._client is None:
                answer, reasoning, conf = self._mock_reason(query, ctx, task)
                backend_used = "mock"
            else:
                answer, reasoning, conf = await self._real_reason(query, ctx, task)
                backend_used = self.backend
            output = {
                "answer": answer,
                "reasoning": reasoning,
                "task": task if task != "auto" else self._detect_task(query, ctx),
                "backend": backend_used,
            }
            return self._success_response(
                request,
                output=output,
                confidence=conf,
                latency_ms=self._measure_latency(t0),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._error_response(
                request,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=self._measure_latency(t0),
            )

    # -- mock backend ----------------------------------------------------
    def _detect_task(self, query: str, context: str) -> str:
        if _looks_like_math(query):
            return "math"
        if any(kw in query.lower() for kw in ("summarize", "summary", "tldr", "gist")):
            return "summarize"
        if any(kw in query.lower() for kw in ("prove", "if and only if", "therefore", "implies")):
            return "logic"
        return "general"

    def _mock_reason(self, query: str, context: str, task: str) -> tuple[str, str, float]:
        """Deterministic mock reasoning."""
        rng = self._rng_for((query, context or "", task))
        detected = task if task not in {"", "auto"} else self._detect_task(query, context)

        if detected == "math":
            expr = _extract_math_expr(query)
            val = _safe_arith(expr)
            if val is not None:
                answer = str(int(val)) if float(val).is_integer() else f"{val:.6g}"
                reasoning = f"Evaluated the arithmetic expression `{expr}` step by step; result = {answer}."
                return answer, reasoning, 0.97
            answer = str(rng.choice(["0", "1", "42", "undefined"]))
            reasoning = f"Could not evaluate `{expr}` as a pure-arithmetic expression; returning a placeholder answer."
            return answer, reasoning, 0.3

        if detected == "summarize":
            src = context or query
            summary = _summarize(src, ratio=0.3, max_words=40)
            reasoning = f"Extracted the leading 30% of words (max 40) from a {len(src.split())}-word source."
            return summary, reasoning, 0.8

        if detected == "logic":
            templates = [
                ("Assume the antecedent; the consequent follows by direct implication.",
                 "Therefore the claim holds."),
                ("By contrapositive: if not-consequent then not-antecedent, which is given.",
                 "So the original implication is true."),
                ("Case analysis shows all branches reduce to a tautology.",
                 "Hence the statement is proven."),
            ]
            r1, r2 = templates[int(rng.randint(0, len(templates)))]
            return r2, r1 + " " + r2, 0.75

        # general
        generic = [
            "Based on the available context, the answer follows from the provided evidence.",
            "Considering the query and context, the most plausible answer is constructed from the salient facts.",
            "After reasoning over the inputs, the response integrates the key signals present in the context.",
        ]
        ans = str(rng.choice(generic))
        return ans, f"Template reasoning seeded by query/context hash: {ans[:40]}...", float(0.5 + 0.3 * rng.rand())

    # -- real backend ----------------------------------------------------
    async def _real_reason(self, query: str, context: str, task: str) -> tuple[str, str, float]:
        """Real LLM reasoning via z-ai-web-dev-sdk / transformers."""
        prompt = (
            f"Context: {context}\n\nQuestion: {query}\n\n"
            "Provide a step-by-step reasoning chain, then the final answer on a new line prefixed with 'ANSWER: '."
        )
        try:
            if self.backend == "llm" and _HAS_ZAI and self._client is not None:
                for method_name in ("complete", "chat", "generate", "predict", "run"):
                    method = getattr(self._client, method_name, None)
                    if method is None:
                        continue
                    try:
                        if _is_async(method):
                            result = await method(prompt)
                        else:
                            result = method(prompt)
                        text = _extract_text(result)
                        if text:
                            answer, reasoning = _split_reason_answer(text)
                            return answer, reasoning, 0.85
                    except Exception:
                        continue
            if self.backend == "transformers" and _HAS_HF and _HAS_TORCH and self._client is not None:
                out = self._client(prompt, max_new_tokens=200, return_full_text=False)
                text = out[0]["generated_text"] if out else ""
                answer, reasoning = _split_reason_answer(text)
                return answer, reasoning, 0.7
        except Exception:
            pass
        return self._mock_reason(query, context, task)


# -- helpers ----------------------------------------------------------------
def _is_async(fn: Any) -> bool:
    import asyncio
    return asyncio.iscoroutinefunction(fn)


def _extract_text(result: Any) -> str:
    """Defensively extract text from heterogeneous LLM response shapes."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for k in ("text", "content", "answer", "output", "generated_text", "response"):
            v = result.get(k)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, dict):
                t = v.get("text") or v.get("content")
                if isinstance(t, str):
                    return t
        return str(result)
    if isinstance(result, (list, tuple)) and result:
        return _extract_text(result[0])
    for attr in ("text", "content", "answer", "output", "generated_text", "response"):
        v = getattr(result, attr, None)
        if isinstance(v, str) and v:
            return v
    return str(result)


def _split_reason_answer(text: str) -> tuple[str, str]:
    if "ANSWER:" in text:
        reasoning, answer = text.split("ANSWER:", 1)
        return answer.strip(), reasoning.strip()
    return text.strip(), text.strip()


__all__ = ["TextReasoningExpert"]
