# SPDX-License-Identifier: MIT
# OmniFlow — Unified training-free omni-modal LLM orchestration.
# Copyright (c) 2025 OmniFlow Contributors
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.

"""Configuration loader for OmniFlow.

Provides :func:`load_config` which materialises an :class:`OmniConfig` from a
YAML file (with environment-variable overrides).  PyYAML is an *optional*
dependency: when it is not installed the loader falls back to a minimal
built-in YAML-subset parser sufficient for flat ``key: value`` documents, and
ultimately to environment-variable binding only.

The :class:`OmniConfig` model itself is defined in
:mod:`omnimodal.core.types` (it is part of the public contract) and is
re-exported here for ergonomic access: ``from omnimodal.core.config import
OmniConfig``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omnimodal.core.types import OmniConfig, _SETTINGS_AVAILABLE

__all__ = ["load_config", "default_config", "OmniConfig"]


# ---------------------------------------------------------------------------
# Optional dependency: PyYAML
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore[import-untyped]

    _YAML_AVAILABLE: bool = True
except Exception:  # pragma: no cover - exercised only without PyYAML
    _YAML_AVAILABLE = False


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    """Parse a *flat* YAML subset when PyYAML is unavailable.

    Supports ``key: value`` lines, ``#`` comments, and blank lines.  Values are
    coerced to ``int``/``float``/``bool``/``None`` when possible.  This is **not**
    a full YAML parser — it exists so the framework remains importable and
    usable in minimal sandboxes that lack PyYAML.
    """

    def _coerce(raw: str) -> Any:
        raw = raw.strip()
        if raw == "":
            return ""
        lowered = raw.lower()
        if lowered in {"true", "yes", "on"}:
            return True
        if lowered in {"false", "no", "off"}:
            return False
        if lowered in {"null", "none", "~"}:
            return None
        try:
            if "." in raw or "e" in lowered:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw.strip().strip('"').strip("'")

    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = _coerce(value)
    return out


def _load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict, using PyYAML when available."""
    raw = Path(path).read_text(encoding="utf-8")
    if _YAML_AVAILABLE:
        loaded = yaml.safe_load(raw) or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Config root in {path!s} must be a YAML mapping, got "
                f"{type(loaded).__name__}"
            )
        return loaded
    return _parse_minimal_yaml(raw)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``OMNIFLOW_*`` environment variables on top of ``data``."""
    merged: dict[str, Any] = dict(data)
    prefix = "OMNIFLOW_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        field = key[len(prefix) :].lower()
        merged[field] = _coerce_env(value)
    return merged


def _coerce_env(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: str | Path | None = None, **overrides: Any) -> OmniConfig:
    """Build an :class:`OmniConfig` from a YAML file + environment + overrides.

    Resolution order (later wins):
        1. Field defaults declared on :class:`OmniConfig`.
        2. Values parsed from ``path`` (when provided and the file exists).
        3. ``OMNIFLOW_*`` environment variables.
        4. Keyword ``overrides`` passed to this function.

    When ``pydantic-settings`` is installed and bound, environment variables
    are also honoured natively by the model; the manual overlay here is a
    belt-and-braces fallback that works with or without that dependency.

    Args:
        path: Optional path to a YAML config file.  Missing files are ignored
            (a warning is **not** raised) so callers can probe for optional
            config files without try/except noise.
        **overrides: Field-name -> value pairs that take precedence over
            everything else.

    Returns:
        A fully validated :class:`OmniConfig` instance.
    """

    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            data = _load_yaml_file(p)
    data = _apply_env_overrides(data)
    # Explicit overrides win.
    data.update(overrides)

    # When pydantic-settings is *not* available, BaseSettings is a plain
    # BaseModel, so passing config via constructor is the only path.  When it
    # *is* available the constructor still works identically.
    if _SETTINGS_AVAILABLE and not data:
        # Let BaseSettings do native env binding when there is no file input.
        return OmniConfig()  # type: ignore[call-arg]
    return OmniConfig(**data)  # type: ignore[call-arg]


def default_config() -> OmniConfig:
    """Return a fresh :class:`OmniConfig` populated entirely from defaults."""
    return OmniConfig()  # type: ignore[call-arg]
