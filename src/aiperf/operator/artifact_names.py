# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve run-specific artifact names from the persisted Kubernetes spec."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.path_safety import safe_read_template_path
from aiperf.kubernetes.spec_converter import (
    DEFAULT_KEY_EXPORT_NAMES,
    KeyExportNames,
    key_export_names,
)


def required_raw_exports(spec: Mapping[str, Any] | None) -> frozenset[str]:
    """Require the configured RAW delivery file independently of summary exports."""
    benchmark = (spec or {}).get("benchmark")
    if not isinstance(benchmark, Mapping):
        return frozenset()
    artifacts = benchmark.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("raw") is not True:
        return frozenset()
    normalized = dict(spec)
    normalized["benchmark"] = dict(benchmark, artifacts=dict(artifacts))
    name = key_export_names(normalized).jsonl_name.removesuffix(".jsonl")
    return frozenset({f"{name}_raw.jsonl"})


def raw_exports_materialized(
    run_dir: Path, spec: Mapping[str, Any] | None = None
) -> bool:
    """Check required RAW files on disk without loading large exports into memory."""
    if spec is None:
        serialized = safe_read_template_path(str(run_dir / "job_spec.json"))
        if serialized is None:
            return True
        try:
            spec = orjson.loads(serialized)
        except orjson.JSONDecodeError:
            return False
        if not isinstance(spec, Mapping):
            return False
    required = required_raw_exports(spec)
    return all(
        (run_dir / name).is_file() or (run_dir / f"{name}.zst").is_file()
        for name in required
    )


def key_export_names_from_run_dir(run_dir: Path) -> KeyExportNames:
    """Return export names selected by this run's persisted ``job_spec.json``."""
    try:
        spec: Any = orjson.loads((run_dir / "job_spec.json").read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return DEFAULT_KEY_EXPORT_NAMES
    return key_export_names(spec if isinstance(spec, dict) else None)


def summary_candidates(run_dir: Path) -> tuple[Path, ...]:
    """Return compressed/raw summary paths, with a legacy-name fallback."""
    configured = key_export_names_from_run_dir(run_dir).json_name
    names = [configured]
    if configured != DEFAULT_KEY_EXPORT_NAMES.json_name:
        names.append(DEFAULT_KEY_EXPORT_NAMES.json_name)
    return tuple(
        candidate
        for name in names
        for candidate in (run_dir / f"{name}.zst", run_dir / name)
    )


def find_summary_path(run_dir: Path) -> Path | None:
    """Return the first materialized summary path for a run."""
    return next((path for path in summary_candidates(run_dir) if path.is_file()), None)
