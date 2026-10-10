# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Harbor image tagging and BuildKit arguments."""

import asyncio
import os
import shlex
from pathlib import Path

import aiofiles
import pytest
from pytest import param

from aiperf.common.constants import IS_WINDOWS

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMON_SCRIPT = shlex.quote(str(REPO_ROOT / "ci/scripts/lib/common.sh"))
DOCKER_SCRIPT = shlex.quote(str(REPO_ROOT / "ci/scripts/lib/docker.sh"))
pytestmark = pytest.mark.skipif(IS_WINDOWS, reason="Build scripts require Bash")


async def _run_shell(
    script: str, overrides: dict[str, str]
) -> tuple[int, bytes, bytes]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AIPERF_", "GITHUB_", "IMAGE_", "BUILD_", "CACHE_"))
        and key not in {"REPO_DIR", "NO_CACHE", "PUSH_IMAGE", "PUSH_CACHE"}
    }
    environment["AIPERF_VERSION"] = "0.13.0"
    environment.update(overrides)
    process = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        script,
        cwd=REPO_ROOT,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    assert process.returncode is not None
    return process.returncode, stdout, stderr


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides, expected",
    [
        param({"AIPERF_BRANCH": "develop"}, "v0.13.0-dev-1ec5", id="develop"),
        param({"AIPERF_BRANCH": "main"}, "v0.13.0-main-1ec5", id="main"),
        param(
            {"AIPERF_BRANCH": "feature/dashboard_ui"},
            "v0.13.0-feature-dashboard-ui-1ec5",
            id="branch-slashes",
        ),
        param({"IMAGE_TAG_PREFIX": "release"}, "v0.13.0-release-1ec5", id="prefix"),
        param(
            {"AIPERF_BRANCH": "develop", "AIPERF_VERSION": "0.14.0"},
            "v0.14.0-dev-1ec5",
            id="version",
        ),
        param(
            {"GITHUB_REF_NAME": "main", "GITHUB_HEAD_REF": "develop"},
            "v0.13.0-dev-1ec5",
            id="pr-head-branch",
        ),
        param(
            {"AIPERF_BRANCH": "develop", "AIPERF_SHA": "", "GITHUB_SHA": "abcd1234"},
            "v0.13.0-dev-abcd",
            id="github-sha",
        ),
        param(
            {"REPO_DIR": "/missing/checkout", "AIPERF_TAG": "custom-tag"},
            "custom-tag",
            id="explicit-tag-without-git",
        ),
    ],
)  # fmt: skip
async def test_resolve_image_tag_overrides_expected(
    overrides: dict[str, str], expected: str
) -> None:
    """Tags derive from the selected source identity and optional overrides."""
    returncode, stdout, stderr = await _run_shell(
        f"source {COMMON_SCRIPT}; aiperf_ci_init; aiperf_ci_resolve_tag",
        {"AIPERF_SHA": "1ec5f974", **overrides},
    )
    assert returncode == 0, stderr.decode()
    assert stdout.decode().strip() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides, error",
    [
        param({"AIPERF_SHA": "bad"}, "Commit SHA", id="short-sha"),
        param({"AIPERF_SHA": "zzzz"}, "Commit SHA", id="non-hex-sha"),
        param({"AIPERF_TAG": "bad/tag"}, "Invalid image tag", id="invalid-tag"),
        param({"IMAGE_TAG_PREFIX": "x" * 130}, "Invalid image tag", id="long-tag"),
    ],
)  # fmt: skip
async def test_resolve_image_tag_invalid_input_rejected(
    overrides: dict[str, str], error: str
) -> None:
    """Invalid tags fail before creating a builder or starting a build."""
    returncode, _, stderr = await _run_shell(
        f"source {COMMON_SCRIPT}; aiperf_ci_init; aiperf_ci_resolve_tag",
        {"AIPERF_SHA": "1ec5f974", **overrides},
    )
    assert returncode != 0
    assert error in stderr.decode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "no_cache, push_cache, expected_flags",
    [
        param("0", "0", ["--cache-from"], id="import-only"),
        param("0", "1", ["--cache-from", "--cache-to"], id="import-export"),
        param("1", "0", ["--no-cache"], id="fresh-local"),
        param("1", "1", ["--no-cache", "--cache-to"], id="fresh-export"),
    ],
)  # fmt: skip
async def test_buildkit_cache_flags_preserve_export_on_clean_build(
    no_cache: str, push_cache: str, expected_flags: list[str]
) -> None:
    """Cache import and export can be independently disabled."""
    returncode, stdout, stderr = await _run_shell(
        f"source {COMMON_SCRIPT}; source {DOCKER_SCRIPT}; "
        "aiperf_ci_init; aiperf_ci_resolve_refs; aiperf_ci_cache_args",
        {
            "AIPERF_SHA": "1ec5f974",
            "AIPERF_BRANCH": "develop",
            "NO_CACHE": no_cache,
            "PUSH_CACHE": push_cache,
            "BUILD_PLATFORM": "linux/amd64",
        },
    )
    assert returncode == 0, stderr.decode()
    arguments = stdout.decode().split("\0")
    assert [
        argument for argument in arguments if argument.startswith("--")
    ] == expected_flags
    if push_cache == "1":
        assert any("aiperf-dev-amd64,mode=max" in argument for argument in arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("push", [param("0", id="load"), param("1", id="push")])
async def test_build_image_script_routes_sources_and_push_options(
    tmp_path: Path, push: str
) -> None:
    """Exercise the whole script without connecting to a Docker daemon."""
    docker = tmp_path / "docker"
    async with aiofiles.open(docker, "w") as script:
        await script.write(
            '#!/bin/bash\nif [[ "$1 $2" == "buildx build" ]]; then\n'
            '  printf "BUILD_ARGS\\0"; printf "%s\\0" "$@"\nfi\n'
        )
    await asyncio.to_thread(docker.chmod, 0o755)
    returncode, stdout, stderr = await _run_shell(
        "./ci/scripts/build-image.sh",
        {
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "AIPERF_BRANCH": "develop",
            "AIPERF_SHA": "1ec5f974",
            "PUSH_IMAGE": push,
            "PYPI_INDEX_URL": "https://pypi.example.com/simple",
            "PYTORCH_INDEX_URL": "https://torch.example.com/cpu",
        },
    )
    assert returncode == 0, stderr.decode()
    arguments = stdout.split(b"BUILD_ARGS\0", 1)[1].decode().split("\0")
    assert ("--push" in arguments) == (push == "1")
    assert ("--load" in arguments) == (push == "0")
    assert ("--cache-to" in arguments) == (push == "1")
    assert "PYPI_INDEX_URL=https://pypi.example.com/simple" in arguments
    assert "PYTORCH_INDEX_URL=https://torch.example.com/cpu" in arguments
    assert "--builder=aiperf-builder" in arguments
    assert "harbor.weizhipin.com/arsenal-ai/aiperf:v0.13.0-dev-1ec5" in arguments
