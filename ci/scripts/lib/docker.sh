#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

_aiperf_ci_has_builder() {
  docker buildx inspect "$1" >/dev/null 2>&1
}

aiperf_ci_ensure_builder() {
  if _aiperf_ci_has_builder "${BUILDX_BUILDER}"; then
    docker buildx inspect "${BUILDX_BUILDER}" --bootstrap >/dev/null
    return
  fi
  docker buildx create --name "${BUILDX_BUILDER}" --driver docker-container \
    --driver-opt "image=${BUILDKIT_IMAGE}" --driver-opt network=host >/dev/null
  docker buildx inspect "${BUILDX_BUILDER}" --bootstrap >/dev/null
}

aiperf_ci_cache_args() {
  if [[ "${NO_CACHE}" == 1 ]]; then
    printf '%s\0' --no-cache
  else
    printf '%s\0' --cache-from "type=registry,ref=${CACHE_REF}"
  fi
  if [[ "${PUSH_CACHE}" == 1 ]]; then
    printf '%s\0' --cache-to "type=registry,ref=${CACHE_REF},mode=max,compression=zstd,oci-mediatypes=true"
  fi
}
