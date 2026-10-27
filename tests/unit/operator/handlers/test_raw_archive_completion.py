# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for long RAW exports and concurrent completion recovery."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
import zstandard as zstd
from aiohttp import web

from aiperf.kubernetes.crd_models import ControllerFetchResult
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.phase import Phase
from aiperf.operator import client_cache as cache
from aiperf.operator.artifact_names import (
    raw_exports_materialized,
    required_raw_exports,
)
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers import _completion_fetch as fetch
from aiperf.operator.handlers import completion, monitor
from aiperf.operator.progress_client import ProgressClient
from aiperf.operator.progress_download import save_zstd_passthrough


@pytest.mark.asyncio
async def test_active_download_survives_explicit_client_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache._reset_for_testing()
    client = AsyncMock()
    monkeypatch.setattr(cache, "ProgressClient", MagicMock(return_value=client))
    key = "ns/job@uid"
    await cache.acquire_progress_client(key)
    cache._shutdown_sent.add(key)
    await cache.close_progress_client(key)
    client.__aexit__.assert_not_awaited()
    assert key in cache._shutdown_sent
    await cache.release_progress_client(key)
    client.__aexit__.assert_awaited_once()
    assert key not in cache._progress_clients
    cache._reset_for_testing()


@pytest.mark.asyncio
async def test_watchdog_leaves_completion_owner_client_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "ns/job@uid"
    cache._shutdown_sent.add(key)
    close = AsyncMock()
    monkeypatch.setattr(monitor, "close_progress_client", close)
    try:
        await monitor._run_worker_and_progress_phase(
            MagicMock(),
            body={},
            status={},
            spec={},
            patch=MagicMock(),
            jobset_status={},
            namespace="ns",
            name="job",
            jobset_name="jobset",
            job_id="job",
            current_phase=Phase.RUNNING,
            key=key,
            sb=MagicMock(),
        )
        close.assert_not_awaited()
    finally:
        cache._shutdown_sent.discard(key)


@pytest.mark.parametrize(
    "prefix,expected",
    [(None, "profile_export_raw.jsonl"), ("custom_raw.jsonl", "custom_raw.jsonl")],
)
def test_required_raw_export_respects_prefix(prefix: str | None, expected: str) -> None:
    artifacts = {"raw": True}
    if prefix is not None:
        artifacts["prefix"] = prefix
    assert required_raw_exports({"benchmark": {"artifacts": artifacts}}) == {expected}
    assert not required_raw_exports({"benchmark": {"artifacts": {"raw": False}}})


@pytest.mark.asyncio
async def test_missing_raw_fetch_uses_sidecar_despite_available_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
    monkeypatch.setattr(K8sEnvironment.PORTS, "RESULTS_SIDECAR", 9091)
    primary = AsyncMock()
    primary.download_all_results.return_value = ["profile_export_aiperf.json"]
    sidecar = AsyncMock()
    sidecar.download_all_results.return_value = ["profile_export_raw.jsonl"]
    context = AsyncMock()
    context.__aenter__.return_value = sidecar
    factory = MagicMock(return_value=context)
    monkeypatch.setattr(fetch, "ProgressClient", factory)
    state = {
        "downloaded": None,
        "checkpoints": None,
        "required_files": {"profile_export_raw.jsonl"},
    }
    await fetch._download_final_and_sidecar(
        progress_client=primary,
        controller_host="host",
        dest_dir=tmp_path,
        state=state,
    )
    factory.assert_called_once_with(port=9091)
    assert "profile_export_raw.jsonl" in state["downloaded"]


@pytest.mark.asyncio
async def test_missing_raw_cannot_return_successful_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
    monkeypatch.setattr(
        K8sEnvironment.PORTS, "RESULTS_SIDECAR", K8sEnvironment.PORTS.API_SERVICE
    )
    primary = AsyncMock()
    primary.download_all_results.return_value = ["profile_export_aiperf.json"]
    state = {
        "metrics": {"metrics": {}},
        "downloaded": None,
        "checkpoints": None,
        "required_files": {"profile_export_raw.jsonl"},
    }
    with pytest.raises(fetch._IncompleteResultsError) as error:
        await fetch._fetch_once_into_state(
            key="ns/job",
            controller_host="host",
            dest_dir=tmp_path,
            progress_client=primary,
            state=state,
        )
    assert error.value.to_fetch_result("job").error


def test_disk_recovery_does_not_promote_summary_without_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
    body = {
        "metadata": {"creationTimestamp": "2026-10-27T00:00:00Z"},
        "spec": {"benchmark": {"artifacts": {"raw": True}}},
    }
    epoch = completion.epoch_key_from_body(body)
    run_dir = completion.run_dir(tmp_path, "ns", "job", epoch)
    run_dir.mkdir(parents=True)
    (run_dir / "profile_export_aiperf.json").write_bytes(b'{"metrics": {"x": 1}}')
    (run_dir / "job_spec.json").write_bytes(orjson.dumps(body["spec"]))
    result = ControllerFetchResult(metrics=None, downloaded=[], error="RAW missing")
    assert (
        completion._recover_result_from_disk(
            body=body, namespace="ns", job_id="job", result=result
        )
        is result
    )
    assert not completion._key_files_materialized("ns", "job", epoch)
    (run_dir / "profile_export_raw.jsonl.zst").write_bytes(
        zstd.ZstdCompressor().compress(b"{}\n")
    )
    assert raw_exports_materialized(run_dir)
    assert completion._key_files_materialized("ns", "job", epoch)


@pytest.mark.asyncio
async def test_truncated_zstd_raw_never_commits_final_file(tmp_path: Path) -> None:
    data = zstd.ZstdCompressor().compress(b'{"sample": 1}\n')[:-2]

    async def chunks(_size: int) -> AsyncIterator[bytes]:
        yield data

    response = MagicMock()
    response.content.iter_chunked = chunks
    target = tmp_path / "profile_export_raw.jsonl"
    with pytest.raises(zstd.ZstdError, match="Incomplete"):
        await save_zstd_passthrough(response, target)
    assert not Path(f"{target}.zst").exists()
    assert not Path(f"{target}.zst.part").exists()


@pytest.mark.asyncio
async def test_partial_export_listing_cannot_be_accepted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    request = AsyncMock(
        return_value={
            "partial": True,
            "failed_exporters": ["RawRecordAggregator"],
            "files": [{"name": "profile_export_aiperf.json", "size": 42}],
        }
    )
    monkeypatch.setattr(ProgressClient, "_request_with_retry", request)
    assert await ProgressClient().get_results_list("controller") is None
    assert "RawRecordAggregator" in caplog.text


@pytest.mark.asyncio
async def test_slow_raw_transfer_survives_cleanup_and_total_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a real HTTP download alive past cleanup and the old total deadline."""
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DOWNLOAD_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "COMPRESS_ON_DISK", True)
    payload = zstd.ZstdCompressor().compress(b'{"sample": "complete"}\n')
    started = asyncio.Event()

    async def serve(request: web.Request) -> web.StreamResponse:
        if request.path.endswith("/list"):
            return web.json_response({"files": [{"name": "profile_export_raw.jsonl"}]})
        response = web.StreamResponse(headers={"Content-Encoding": "zstd"})
        await response.prepare(request)
        for offset in range(0, len(payload), 5):
            await response.write(payload[offset : offset + 5])
            started.set()
            delay = asyncio.Event()
            timer = asyncio.get_running_loop().call_later(0.08, delay.set)
            try:
                await delay.wait()
            finally:
                timer.cancel()
        return response

    app = web.Application()
    app.router.add_get("/api/results/{tail:.*}", serve)
    runner = web.AppRunner(app)
    await runner.setup()
    server = await asyncio.get_running_loop().create_server(
        runner.server, "127.0.0.1", 0
    )
    key = "diagnostic/slow-raw"
    client = await cache.acquire_progress_client(key)
    client._port = server.sockets[0].getsockname()[1]
    session = client._session
    download = asyncio.create_task(client.download_all_results("127.0.0.1", tmp_path))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        await cache.close_progress_client(key)
        assert not session.closed
        assert await asyncio.wait_for(download, timeout=5) == [
            "profile_export_raw.jsonl"
        ]
        assert (
            zstd.ZstdDecompressor().decompress(
                (tmp_path / "profile_export_raw.jsonl.zst").read_bytes()
            )
            == b'{"sample": "complete"}\n'
        )
    finally:
        download.cancel()
        await asyncio.gather(download, return_exceptions=True)
        await cache.release_progress_client(key)
        server.close()
        await server.wait_closed()
        await runner.cleanup()
    assert session.closed
