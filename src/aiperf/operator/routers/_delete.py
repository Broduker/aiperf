# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Kubernetes deletion helper for Dashboard mutation routes."""

from __future__ import annotations

from fastapi import HTTPException
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException


async def delete_live_custom_resource(
    api: ApiClient,
    *,
    group: str,
    version: str,
    plural: str,
    kind: str,
    namespace: str,
    name: str,
) -> None:
    """Delete one live CR with identity and version preconditions.

    The preconditions prevent a stale Dashboard click from deleting a
    replacement resource that reused the same namespace/name after the page
    was loaded. Foreground propagation lets Kubernetes clean up owned workload
    resources before the CR disappears.
    """
    custom = client.CustomObjectsApi(api)
    try:
        raw = await custom.get_namespaced_custom_object(
            group=group,
            version=version,
            plural=plural,
            namespace=namespace,
            name=name,
        )
    except ApiException as exc:
        if exc.status == 404:
            raise HTTPException(404, f"{kind} {namespace}/{name} not found") from exc
        raise

    metadata = raw.get("metadata") or {}
    uid = metadata.get("uid")
    resource_version = metadata.get("resourceVersion")
    if not uid or not resource_version:
        raise HTTPException(
            409,
            f"Cannot safely delete {kind} {namespace}/{name}: "
            "the Kubernetes resource identity is incomplete.",
        )

    try:
        await custom.delete_namespaced_custom_object(
            group=group,
            version=version,
            plural=plural,
            namespace=namespace,
            name=name,
            body=client.V1DeleteOptions(
                propagation_policy="Foreground",
                preconditions=client.V1Preconditions(
                    uid=uid,
                    resource_version=resource_version,
                ),
            ),
        )
    except ApiException as exc:
        if exc.status == 404:
            raise HTTPException(404, f"{kind} {namespace}/{name} not found") from exc
        raise
