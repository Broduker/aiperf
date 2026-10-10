# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static contracts for the Dashboard benchmark submission page."""

from __future__ import annotations

from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[2] / "src" / "aiperf" / "operator" / "ui"
SUBMIT_PATH = UI_ROOT / "pages" / "submit-job.js"


def test_submit_page_validates_before_mutating() -> None:
    source = SUBMIT_PATH.read_text()

    assert "api.validateManifest(manifest)" in source
    assert "if (!result.passed) return;" in source
    assert "api.createJob(manifest)" in source
    assert "api.createSweep(manifest)" in source
    assert "manifest.kind === 'AIPerfSweep'" in source
    assert "buildSweepPath(created)" in source
    assert "isTokenRequiredError(error)" in source
    assert 'testid="submit-job-yaml"' in source


def test_submit_page_does_not_offer_secret_input() -> None:
    source = SUBMIT_PATH.read_text()

    assert "api_key" not in source.lower()
    assert "Do not put API keys" in source


def test_submit_page_escapes_inline_code_in_template_literal() -> None:
    source = SUBMIT_PATH.read_text()

    assert "<code>AIPerfJob</code>" in source
    assert "<code>kubectl apply</code>" in source
    assert "same `AIPerfJob` shape" not in source
