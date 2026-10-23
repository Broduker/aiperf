# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dashboard pure helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.ui.node_utils import run_node

DASHBOARD_HELPERS_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "dashboard-helpers.js"
)


def test_recent_jobs_includes_failed_jobs_in_created_order() -> None:
    script = f"""
        import {{ recentJobs }} from {DASHBOARD_HELPERS_PATH.as_uri()!r};
        const jobs = [
          {{ name: 'old-completed', phase: 'Succeeded', created: '2026-05-17T10:00:00Z' }},
          {{ name: 'new-failed', phase: 'Failed', created: '2026-05-17T12:00:00Z' }},
          {{ name: 'running', phase: 'Running', created: '2026-05-17T13:00:00Z' }},
          {{ name: 'new-error', phase: 'Error', created: '2026-05-17T11:00:00Z' }},
        ];
        console.log(JSON.stringify(recentJobs(jobs).map(j => j.name)));
    """

    assert json.loads(run_node(script)) == [
        "new-failed",
        "new-error",
        "old-completed",
    ]


def test_progress_preserves_small_percentages_and_does_not_round_up_completion() -> (
    None
):
    script = f"""
        import {{ jobProgressPercent }} from {DASHBOARD_HELPERS_PATH.as_uri()!r};
        console.log(JSON.stringify([null, 0, 0.01, 0.1, 0.4, 99.99, 100, NaN].map(jobProgressPercent)));
    """
    assert json.loads(run_node(script)) == [None, 0, 0.01, 0.1, 0.4, 99.9, 100, None]
