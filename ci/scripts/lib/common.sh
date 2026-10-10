#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

aiperf_ci_init() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
  export SCRIPT_DIR REPO_DIR

  HARBOR_REGISTRY="${HARBOR_REGISTRY:-harbor.weizhipin.com}"
  AIPERF_IMAGE="${AIPERF_IMAGE:-${HARBOR_REGISTRY}/arsenal-ai/aiperf}"
  PYTHON_IMAGE="${PYTHON_IMAGE:-${HARBOR_REGISTRY}/arsenal-ai/python:3.12-slim}"
  UV_IMAGE="${UV_IMAGE:-${HARBOR_REGISTRY}/arsenal-ai/uv:latest}"
  PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://mirrors.sjtug.sjtu.edu.cn/pytorch-wheels/cpu}"
  DEBIAN_MIRROR="${DEBIAN_MIRROR:-mirrors.aliyun.com}"
  FFMPEG_SOURCE_URL="${FFMPEG_SOURCE_URL:-https://ffmpeg.org/releases}"
  BUILD_PLATFORM="${BUILD_PLATFORM:-linux/amd64}"
  BUILD_PROGRESS="${BUILD_PROGRESS:-plain}"
  BUILDX_BUILDER="${BUILDX_BUILDER:-aiperf-builder}"
  BUILDKIT_IMAGE="${BUILDKIT_IMAGE:-${HARBOR_REGISTRY}/arsenal-ai/moby/buildkit:buildx-stable-1}"
  CACHE_IMAGE="${CACHE_IMAGE:-${HARBOR_REGISTRY}/arsenal-ai/aiperf-build-cache}"
  CACHE_SCOPE="${CACHE_SCOPE:-aiperf}"
  NO_CACHE="${NO_CACHE:-0}"
  PUSH_IMAGE="${PUSH_IMAGE:-0}"
  PUSH_CACHE="${PUSH_CACHE:-${PUSH_IMAGE}}"
  ENABLE_FFMPEG="${ENABLE_FFMPEG:-1}"
  export HARBOR_REGISTRY AIPERF_IMAGE PYTHON_IMAGE UV_IMAGE PYPI_INDEX_URL PYTORCH_INDEX_URL
  export DEBIAN_MIRROR FFMPEG_SOURCE_URL BUILD_PLATFORM BUILD_PROGRESS
  export BUILDX_BUILDER BUILDKIT_IMAGE CACHE_IMAGE CACHE_SCOPE NO_CACHE PUSH_IMAGE PUSH_CACHE ENABLE_FFMPEG
}

aiperf_ci_branch() {
  local branch
  branch="${AIPERF_BRANCH:-${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}}"
  if [[ -z "${branch}" ]]; then
    branch="$(git -C "${REPO_DIR}" symbolic-ref --short HEAD 2>/dev/null || true)"
  fi
  branch="${branch:-detached}"
  printf '%s\n' "${branch}"
}

aiperf_ci_prefix() {
  local branch prefix
  branch="$(aiperf_ci_branch)"
  prefix="${IMAGE_TAG_PREFIX:-${branch}}"
  [[ "${prefix}" == "develop" ]] && prefix=dev
  prefix="$(printf '%s' "${prefix}" | tr '/_' '--' | tr -cd '[:alnum:].-')"
  prefix="${prefix#.}"
  prefix="${prefix#-}"
  prefix="${prefix%.}"
  prefix="${prefix%-}"
  printf '%s\n' "${prefix:-dev}"
}

aiperf_ci_version() {
  local version
  version="${AIPERF_VERSION:-$(sed -n 's/^version = "\([^"]*\)"$/\1/p' "${REPO_DIR}/pyproject.toml" | head -n 1)}"
  [[ -n "${version}" ]] || { echo "Unable to resolve AIPerf version" >&2; return 1; }
  [[ "${version}" == v* ]] || version="v${version}"
  printf '%s\n' "${version}"
}

aiperf_ci_resolve_tag() {
  local sha prefix version tag
  tag="${AIPERF_TAG:-}"
  if [[ -z "${tag}" ]]; then
    sha="${AIPERF_SHA:-${GITHUB_SHA:-$(git -C "${REPO_DIR}" rev-parse HEAD)}}"
    [[ "${sha}" =~ ^[a-fA-F0-9]{4,40}$ ]] || { echo "Commit SHA must contain 4 to 40 hexadecimal characters" >&2; return 1; }
    prefix="$(aiperf_ci_prefix)"
    version="$(aiperf_ci_version)"
    tag="${version}-${prefix}-${sha:0:4}"
  fi
  [[ "${tag}" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ ]] || { echo "Invalid image tag (maximum 128 characters): ${tag}" >&2; return 1; }
  printf '%s\n' "${tag}"
}

aiperf_ci_resolve_refs() {
  AIPERF_TAG="$(aiperf_ci_resolve_tag)"
  AIPERF_IMAGE_REF="${AIPERF_IMAGE}:${AIPERF_TAG}"
  CACHE_REF="${CACHE_IMAGE}:${CACHE_SCOPE}-$(aiperf_ci_prefix)-${BUILD_PLATFORM##*/}"
  export AIPERF_TAG AIPERF_IMAGE_REF CACHE_REF
}
