#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"

aiperf_ci_init
aiperf_ci_resolve_refs
aiperf_ci_ensure_builder
cd "${REPO_DIR}"

CACHE_ARGS=()
while IFS= read -r -d '' argument; do
  CACHE_ARGS+=("${argument}")
done < <(aiperf_ci_cache_args)

echo "Building ${AIPERF_IMAGE_REF}"
echo "Branch: $(aiperf_ci_branch)"
echo "Version: $(aiperf_ci_version)"
echo "Platform: ${BUILD_PLATFORM}"
echo "Python image: ${PYTHON_IMAGE}"
echo "uv image: ${UV_IMAGE}"
echo "PyPI index: ${PYPI_INDEX_URL}"
echo "Cache: ${CACHE_REF}"

PUSH_ARGS=()
if [[ "${PUSH_IMAGE}" == 1 ]]; then
  PUSH_ARGS+=(--push)
else
  PUSH_ARGS+=(--load)
fi

docker buildx build \
  --builder="${BUILDX_BUILDER}" \
  --progress="${BUILD_PROGRESS}" \
  --network=host \
  --platform="${BUILD_PLATFORM}" \
  --target=runtime \
  "${PUSH_ARGS[@]}" \
  "${CACHE_ARGS[@]}" \
  --build-arg "PYTHON_IMAGE=${PYTHON_IMAGE}" \
  --build-arg "UV_IMAGE=${UV_IMAGE}" \
  --build-arg "PYPI_INDEX_URL=${PYPI_INDEX_URL}" \
  --build-arg "PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}" \
  --build-arg "DEBIAN_MIRROR=${DEBIAN_MIRROR}" \
  --build-arg "FFMPEG_SOURCE_URL=${FFMPEG_SOURCE_URL}" \
  --build-arg "ENABLE_FFMPEG=${ENABLE_FFMPEG}" \
  --tag "${AIPERF_IMAGE_REF}" \
  --file Dockerfile \
  .

echo "Image ready: ${AIPERF_IMAGE_REF}"
