"""Worker dataset-client initialization is serialized and idempotent."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import MemoryMapFormat
from aiperf.common.models.dataset_models import MemoryMapClientMetadata
from aiperf.workers.worker import Worker


class _FakeClient:
    """Lifecycle double used to count mmap client openings and closures."""

    instances: list["_FakeClient"] = []

    def __init__(self, client_metadata: MemoryMapClientMetadata) -> None:
        self.client_metadata = client_metadata
        self.initialize_count = 0
        self.stop_count = 0
        self.__class__.instances.append(self)

    async def initialize(self) -> None:
        self.initialize_count += 1
        await asyncio.sleep(0)

    async def stop(self) -> None:
        self.stop_count += 1


def _metadata(path: str) -> MemoryMapClientMetadata:
    return MemoryMapClientMetadata(
        format=MemoryMapFormat.CONVERSATION,
        data_file_path=f"/datasets/{path}/dataset.dat",
        index_file_path=f"/datasets/{path}/index.dat",
        conversation_count=1,
        total_size_bytes=8,
    )


def _worker(monkeypatch: pytest.MonkeyPatch) -> Worker:
    worker = Worker.__new__(Worker)
    worker._dataset_client = None
    worker._dataset_client_key = None
    worker._dataset_client_lock = asyncio.Lock()
    worker._dataset_configured_event = asyncio.Event()
    worker._is_payload_bytes = False
    worker.run = SimpleNamespace(
        cfg=SimpleNamespace(get_cache_bust_target=lambda: None),
    )
    worker.debug = lambda message: None
    worker._is_group_managed_mode = lambda: False
    worker._mark_worker_ready = AsyncMock()
    monkeypatch.setattr(
        "aiperf.workers.worker.plugins.get_class", lambda *_args: _FakeClient
    )
    _FakeClient.instances.clear()
    return worker


@pytest.mark.asyncio
async def test_same_dataset_notifications_open_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker(monkeypatch)
    metadata = _metadata("class1")

    await asyncio.gather(
        Worker._open_dataset_client(worker, metadata),
        Worker._open_dataset_client(worker, metadata),
    )

    assert len(_FakeClient.instances) == 1
    assert _FakeClient.instances[0].initialize_count == 1
    assert _FakeClient.instances[0].stop_count == 0


@pytest.mark.asyncio
async def test_changed_dataset_notification_closes_previous_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker(monkeypatch)

    await Worker._open_dataset_client(worker, _metadata("class1"))
    await Worker._open_dataset_client(worker, _metadata("daily-coding"))

    assert len(_FakeClient.instances) == 2
    assert _FakeClient.instances[0].stop_count == 1
    assert _FakeClient.instances[1].stop_count == 0
