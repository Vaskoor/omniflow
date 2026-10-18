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
"""CrossModalMemory — text-centric cross-modal memory store.

The store keeps :class:`MemoryEntry` objects (text + modality tag +
source-expert + optional embedding id) and supports three retrieval modes:

1. **TF-IDF + cosine similarity** (default; pure numpy) — fast, deterministic,
   works offline.
2. **Keyword overlap (Jaccard)** — used as a tie-breaker / sanity check.
3. **Dense embeddings** via ``sentence-transformers`` (guarded) — set
   ``use_real_embeddings=True`` and have ``sentence-transformers`` installed;
   falls back to TF-IDF if unavailable.

Dense embeddings are stored out-of-band (``MemoryEntry`` only carries an
``embedding_id`` field per the contract); the store maintains a private
``{entry_id -> np.ndarray}`` map.

All retrieval returns ``list[MemoryEntry]``; ``get_context(query)`` flattens
the top-k entries into a single string usable as LLM context.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Optional

import numpy as np

try:
    from omnimodal.core.types import (  # type: ignore[attr-defined]
        ExpertType,
        MemoryEntry,
        ModalityType,
    )
except ImportError:
    from omnimodal.experts.base import (
        ExpertType,
        MemoryEntry,
        ModalityType,
    )

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _coerce_memory_entry(entry: Any) -> MemoryEntry:
    """Duck-type any input into a :class:`MemoryEntry`.

    Accepts:

    * a :class:`MemoryEntry` (returned as-is)
    * a ``dict`` with the contract fields (filtered + validated)
    * any object exposing the same attributes (duck-typed into a new entry)
    """
    if isinstance(entry, MemoryEntry):
        return entry
    if isinstance(entry, dict):
        # filter to known fields
        known = {"id", "text", "modality", "source_expert", "timestamp", "embedding_id"}
        data = {k: v for k, v in entry.items() if k in known}
        if "text" not in data:
            data["text"] = str(entry.get("text", ""))
        if "modality" not in data:
            data["modality"] = ModalityType.TEXT
        if "source_expert" not in data:
            data["source_expert"] = ExpertType.TEXT_REASON
        return MemoryEntry.model_validate(data)
    # duck-typed object
    text = getattr(entry, "text", "") or ""
    modality = getattr(entry, "modality", ModalityType.TEXT) or ModalityType.TEXT
    source = getattr(entry, "source_expert", None) or getattr(entry, "source", None) or ExpertType.TEXT_REASON
    eid = getattr(entry, "id", None) or uuid.uuid4().hex
    ts = getattr(entry, "timestamp", None) or time.time()
    emb_id = getattr(entry, "embedding_id", None)
    return MemoryEntry(
        id=eid,
        text=text,
        modality=modality,
        source_expert=source,
        timestamp=ts,
        embedding_id=emb_id,
    )


class CrossModalMemory:
    """Text-centric cross-modal memory store.

    Parameters
    ----------
    use_real_embeddings
        If True and ``sentence-transformers`` is importable, encode entries
        with a real dense model and use cosine similarity on embeddings for
        retrieval. Otherwise (or if the library is unavailable) the store
        uses pure-numpy TF-IDF.
    embedding_model_name
        Model name passed to ``SentenceTransformer``. Defaults to a small
        all-MiniLM model.
    max_entries
        Soft cap; once exceeded the oldest entries are evicted (FIFO).
    """

    def __init__(
        self,
        *,
        use_real_embeddings: bool = False,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_entries: int = 1000,
        seed: int = 42,
    ):
        self.use_real_embeddings = use_real_embeddings and _HAS_ST
        self.embedding_model_name = embedding_model_name
        self.max_entries = max_entries
        self.seed = seed
        self._entries: list[MemoryEntry] = []
        self._id_index: dict[str, int] = {}
        self._vocab: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []  # term-frequency per entry
        self._df: dict[str, int] = {}  # document-frequency
        self._idf: Optional[np.ndarray] = None
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._embeddings: dict[str, np.ndarray] = {}
        self._encoder: Any = None
        if self.use_real_embeddings:
            try:
                self._encoder = SentenceTransformer(self.embedding_model_name)
            except Exception:
                self._encoder = None
                self.use_real_embeddings = False

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def store(self, entry: MemoryEntry | dict[str, Any] | Any) -> str:
        """Add an entry to memory; returns the entry id.

        Accepts a :class:`MemoryEntry`, a ``dict`` of contract fields, or
        any duck-typed object exposing ``.text`` / ``.modality`` /
        ``.source_expert``.
        """
        e = _coerce_memory_entry(entry)
        if not e.id:
            e.id = uuid.uuid4().hex
        if not e.text:
            e.text = ""
        # Evict if at capacity (FIFO)
        while len(self._entries) >= self.max_entries:
            evicted = self._entries.pop(0)
            self._id_index.pop(evicted.id, None)
            self._tf.pop(0, None)
            self._embeddings.pop(evicted.id, None)
        # Reindex id positions
        self._id_index = {ent.id: i for i, ent in enumerate(self._entries)}
        self._entries.append(e)
        self._id_index[e.id] = len(self._entries) - 1
        # Update TF and DF
        toks = _tokenize(e.text)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        self._tf.append(tf)
        for term in tf:
            self._df[term] = self._df.get(term, 0) + 1
        # Mark TF-IDF cache dirty
        self._idf = None
        self._tfidf_matrix = None
        # If using real embeddings, populate the out-of-band store
        if self.use_real_embeddings and self._encoder is not None and e.id not in self._embeddings:
            try:
                vec = self._encoder.encode([e.text], normalize_embeddings=True)
                self._embeddings[e.id] = np.asarray(vec[0], dtype=float)
                e.embedding_id = e.id
            except Exception:
                pass
        return e.id

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        idx = self._id_index.get(entry_id)
        if idx is None:
            return None
        return self._entries[idx]

    def all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._id_index.clear()
        self._vocab.clear()
        self._tf.clear()
        self._df.clear()
        self._idf = None
        self._tfidf_matrix = None
        self._embeddings.clear()

    def size(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """Return the top-k entries most similar to ``query``."""
        if not self._entries or k <= 0:
            return []
        k = min(k, len(self._entries))
        if self.use_real_embeddings and self._encoder is not None:
            return self._retrieve_dense(query, k)
        return self._retrieve_tfidf(query, k)

    def _retrieve_tfidf(self, query: str, k: int) -> list[MemoryEntry]:
        scores = self._tfidf_scores(query)
        if scores is None:
            scores = self._jaccard_scores(query)
        order = np.argsort(scores)[::-1][:k]
        return [self._entries[i] for i in order]

    def _retrieve_dense(self, query: str, k: int) -> list[MemoryEntry]:
        try:
            qv = np.asarray(self._encoder.encode([query], normalize_embeddings=True)[0], dtype=float)
        except Exception:
            return self._retrieve_tfidf(query, k)
        sims: list[float] = []
        for e in self._entries:
            vec = self._embeddings.get(e.id)
            if vec is None:
                # compute on the fly and cache
                try:
                    v = self._encoder.encode([e.text], normalize_embeddings=True)
                    vec = np.asarray(v[0], dtype=float)
                    self._embeddings[e.id] = vec
                    e.embedding_id = e.id
                except Exception:
                    vec = None
            if vec is not None and vec.size > 0:
                v = vec / (np.linalg.norm(vec) + 1e-9)
                sims.append(float(np.dot(qv, v)))
            else:
                sims.append(0.0)
        order = np.argsort(np.array(sims))[::-1][:k]
        return [self._entries[i] for i in order]

    def _jaccard_scores(self, query: str) -> np.ndarray:
        qset = set(_tokenize(query))
        if not qset:
            return np.zeros(len(self._entries), dtype=float)
        out = np.zeros(len(self._entries), dtype=float)
        for i, tf in enumerate(self._tf):
            dset = set(tf.keys())
            if not dset:
                continue
            inter = len(qset & dset)
            union = len(qset | dset)
            out[i] = inter / union if union else 0.0
        return out

    def _tfidf_scores(self, query: str) -> Optional[np.ndarray]:
        """Cosine similarity between the query TF-IDF vector and stored entries."""
        if not self._entries:
            return None
        self._rebuild_cache()
        if self._tfidf_matrix is None or self._idf is None:
            return None
        q_toks = _tokenize(query)
        if not q_toks:
            return np.zeros(len(self._entries), dtype=float)
        q_tf: dict[str, int] = {}
        for t in q_toks:
            q_tf[t] = q_tf.get(t, 0) + 1
        q_vec = np.zeros(len(self._vocab), dtype=float)
        for term, cnt in q_tf.items():
            j = self._vocab.get(term)
            if j is None:
                continue
            tf = float(cnt)
            idf = float(self._idf[j])
            q_vec[j] = tf * idf
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-9:
            return np.zeros(len(self._entries), dtype=float)
        q_vec = q_vec / q_norm
        return self._tfidf_matrix @ q_vec

    def _rebuild_cache(self) -> None:
        if self._tfidf_matrix is not None:
            return
        self._vocab = {}
        for tf in self._tf:
            for term in tf:
                if term not in self._vocab:
                    self._vocab[term] = len(self._vocab)
        n = len(self._entries)
        V = len(self._vocab)
        if V == 0:
            self._idf = np.zeros(0, dtype=float)
            self._tfidf_matrix = np.zeros((n, 0), dtype=float)
            return
        idf = np.zeros(V, dtype=float)
        for term, j in self._vocab.items():
            df = self._df.get(term, 0)
            idf[j] = float(np.log((1.0 + n) / (1.0 + df)) + 1.0)
        self._idf = idf
        matrix = np.zeros((n, V), dtype=float)
        for i, tf in enumerate(self._tf):
            for term, cnt in tf.items():
                j = self._vocab[term]
                matrix[i, j] = float(cnt) * idf[j]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        self._tfidf_matrix = matrix / norms

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------
    def get_context(self, query: str, k: int = 5, max_chars: int = 4000) -> str:
        """Build a text context string from the top-k memory entries."""
        entries = self.retrieve(query, k=k)
        if not entries:
            return ""
        parts: list[str] = []
        total = 0
        for e in entries:
            mod = e.modality.value if hasattr(e.modality, "value") else str(e.modality)
            src = e.source_expert.value if hasattr(e.source_expert, "value") else str(e.source_expert)
            header = f"[{src}:{mod}]"
            chunk = f"{header} {e.text}"
            if total + len(chunk) > max_chars:
                remaining = max_chars - total
                if remaining > 40:
                    parts.append(chunk[:remaining] + "...")
                break
            parts.append(chunk)
            total += len(chunk) + 1
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._entries),
            "vocab_size": len(self._vocab),
            "use_real_embeddings": self.use_real_embeddings,
            "encoder": self.embedding_model_name if self.use_real_embeddings else None,
            "max_entries": self.max_entries,
            "embeddings_cached": len(self._embeddings),
        }


__all__ = ["CrossModalMemory"]
