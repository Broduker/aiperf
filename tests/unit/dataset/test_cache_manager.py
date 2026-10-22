"""Historical result recovery uses payload identity and disk-backed accounting."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import orjson
import pytest
import zstandard

from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.config.dataset import FileDataset
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.cache_manager import CacheManager, _successful_payload
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.loader.base_loader import _default_test_run
from aiperf.dataset.loader.single_turn import SingleTurnDatasetLoader
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClient,
)
from aiperf.dataset.payload_formatting import format_conversation_payloads


def make_run(tmp_path: Path, sources: list[Path]) -> BenchmarkRun:
    run = _default_test_run()
    source = tmp_path / "input.jsonl"
    source.touch()
    run.cfg.datasets = [
        FileDataset(
            name="default",
            type="file",
            format="single_turn",
            path=source,
            resume_from=sources,
        )
    ]
    run.cfg.artifacts.dir = tmp_path / "new-run"
    run.artifact_dir = run.cfg.artifacts.dir
    return run


def conversation(session_id: str = "sample-1", text: str = "hello") -> Conversation:
    return Conversation(
        session_id=session_id,
        turns=[Turn(texts=[Text(contents=[text])], model="test-model")],
    )


def raw_record(run: BenchmarkRun, sample: Conversation) -> dict:
    payload = next(
        format_conversation_payloads([sample], ModelEndpointInfo.from_run(run))
    )[2]
    return {
        "metadata": {
            "conversation_id": sample.session_id,
            "turn_index": 0,
            "benchmark_phase": "profiling",
        },
        "status": 200,
        "payload": payload,
        "responses": [
            {
                "text": orjson.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "message": {"content": "answer"},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).decode()
            }
        ],
    }


@pytest.mark.parametrize("compressed", [False, True])
def test_recovery_matches_payload_success_wins_and_keeps_history(
    tmp_path: Path, compressed: bool
) -> None:
    history = tmp_path / (
        "raw_records_old.jsonl.zst" if compressed else "raw_records_old.jsonl"
    )
    run = make_run(tmp_path, [history])
    good = raw_record(run, conversation())
    bad = {**good, "error": {"message": "later request failed"}}
    data = b"\n".join(orjson.dumps(record) for record in [good, bad, good]) + b"\n"
    history.write_bytes(
        zstandard.ZstdCompressor().compress(data) if compressed else data
    )
    original = history.read_bytes()
    cache = CacheManager(run)
    try:
        samples = [
            conversation(),
            conversation("sample-2"),
            conversation("sample-3", "changed"),
        ]
        assert [sample.session_id for sample in cache.pending(samples)] == [
            "sample-2",
            "sample-3",
        ]
        cache.save_report()
        assert cache.report.successful_records == 1
        assert cache.report.cached_samples == 1
        assert cache.report.pending_samples == 2
        assert cache.report.ignored_records == 1
        assert (
            orjson.loads((cache.directory / "reused.jsonl").read_bytes())[
                "conversation_id"
            ]
            == "sample-1"
        )
        assert history.read_bytes() == original
    finally:
        cache.close()


@pytest.mark.parametrize("mutation", ["prompt", "model", "max_tokens", "extra"])
def test_changed_request_is_never_reused(tmp_path: Path, mutation: str) -> None:
    history = tmp_path / "raw_records_old.jsonl"
    run = make_run(tmp_path, [history])
    record = raw_record(run, conversation())
    if mutation == "prompt":
        record["payload"]["messages"] = [{"role": "user", "content": "other prompt"}]
    else:
        record["payload"][mutation] = "changed"
    history.write_bytes(orjson.dumps(record) + b"\n")
    cache = CacheManager(run)
    try:
        assert len(list(cache.pending([conversation()]))) == 1
        assert cache.report.cached_samples == 0
    finally:
        cache.close()


@pytest.mark.parametrize(
    "kind",
    [
        "error",
        "status",
        "empty",
        "length",
        "warmup",
        "turn",
        "missing_payload",
        "malformed_response",
    ],
)
def test_unsuccessful_or_unsupported_record_is_retried(
    tmp_path: Path, kind: str
) -> None:
    run = make_run(tmp_path, [])
    record = raw_record(run, conversation())
    if kind == "error":
        record["error"] = {"message": "failed"}
    elif kind == "status":
        record["status"] = 500
    elif kind == "warmup":
        record["metadata"]["benchmark_phase"] = "warmup"
    elif kind == "turn":
        record["metadata"]["turn_index"] = 1
    elif kind == "missing_payload":
        record.pop("payload")
    elif kind == "malformed_response":
        record["responses"] = [{"text": "broken"}]
    else:
        record["responses"] = [
            {
                "text": orjson.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "" if kind == "empty" else "truncated"
                                },
                                "finish_reason": "length"
                                if kind == "length"
                                else "stop",
                            }
                        ]
                    }
                ).decode()
            }
        ]
    assert _successful_payload(record) is None


def test_streaming_response_requires_terminal_stop_and_content(tmp_path: Path) -> None:
    record = raw_record(make_run(tmp_path, []), conversation())
    record["responses"] = [
        {
            "data": orjson.dumps(
                {"choices": [{"index": 0, "delta": {"content": "answer"}}]}
            ).decode()
        },
        {
            "data": orjson.dumps(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            ).decode()
        },
        {"data": "[DONE]"},
    ]
    assert _successful_payload(record) == record["payload"]
    record["responses"].pop(1)
    assert _successful_payload(record) is None


def test_partial_tail_is_not_success_and_malformed_interior_fails(
    tmp_path: Path,
) -> None:
    history = tmp_path / "raw_records_old.jsonl"
    run = make_run(tmp_path, [history])
    history.write_bytes(orjson.dumps(raw_record(run, conversation())))
    cache = CacheManager(run)
    try:
        assert cache.report.incomplete_tails == 1
        assert len(list(cache.pending([conversation()]))) == 1
    finally:
        cache.close()
    history.write_bytes(b"{broken}\n")
    with pytest.raises(ValueError, match="Invalid RAW JSON"):
        CacheManager(run)


def test_duplicate_input_ids_rejected(tmp_path: Path) -> None:
    history = tmp_path / "raw_records_old.jsonl"
    history.write_bytes(b"")
    cache = CacheManager(make_run(tmp_path, [history]))
    try:
        with pytest.raises(ValueError, match="Duplicate session_id"):
            list(cache.pending([conversation(), conversation()]))
    finally:
        cache.close()


def test_missing_explicit_id_rejected_by_loader(tmp_path: Path) -> None:
    run = make_run(tmp_path, [tmp_path / "old"])
    run.cfg.datasets[0].path.write_bytes(b'{"text":"hello"}\n')
    loader = SingleTurnDatasetLoader(run=run, filename=run.cfg.datasets[0].path)
    with pytest.raises(ValueError, match="explicit non-empty session_id"):
        loader.load_dataset()


def test_resume_requires_supported_dataset_and_single_phase(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path-backed single_turn"):
        FileDataset(
            name="default",
            type="file",
            records=[{"text": "hello"}],
            resume_from=[tmp_path],
        )
    run = make_run(tmp_path, [tmp_path / "old"])
    run.cfg.phases.append(run.cfg.phases[0].model_copy(update={"name": "second"}))
    with pytest.raises(ValueError, match="one dataset and one profiling phase"):
        run.cfg.validate_result_cache_resume()


@pytest.mark.parametrize("all_cached", [True, False])
@pytest.mark.parametrize("compressed", [False, True])
async def test_manager_writes_only_pending_and_handles_empty_mmap(
    tmp_path: Path, all_cached: bool, compressed: bool
) -> None:
    history = tmp_path / "raw_records_old.jsonl"
    run = make_run(tmp_path, [history])
    run.cfg.datasets[0].path.write_bytes(
        b'{"session_id":"sample-1","text":"hello"}\n'
        b'{"session_id":"sample-2","text":"hello"}\n'
    )
    samples = list(CustomDatasetComposer(run=run, tokenizer=None).iter_conversations())
    history.write_bytes(
        b"".join(
            orjson.dumps(raw_record(run, sample)) + b"\n"
            for sample in (samples if all_cached else samples[:1])
        )
    )
    manager = DatasetManager(run=run)
    manager._compress_only = compressed
    manager.publish = AsyncMock()
    manager._cache_manager = CacheManager(run)
    try:
        await manager._configure_dataset()
        metadata = manager._backing_store.get_client_metadata()
        assert metadata.conversation_count == (0 if all_cached else 1)
        assert [
            sample.conversation_id for sample in manager.dataset_metadata.conversations
        ] == ([] if all_cached else ["sample-2"])
        assert manager._should_skip_inputs_json()
        assert manager._try_cache_lookup() is None
        data_path = metadata.data_file_path
        index_path = metadata.index_file_path
        if compressed:
            data_path = tmp_path / "dataset.dat"
            index_path = tmp_path / "index.dat"
            for compressed_path, target in (
                (metadata.compressed_data_file_path, data_path),
                (metadata.compressed_index_file_path, index_path),
            ):
                with zstandard.open(compressed_path, "rb") as compressed_file:
                    target.write_bytes(compressed_file.read())
        with MemoryMapDatasetClient(data_path, index_path) as client:
            assert client.index.conversation_ids == ([] if all_cached else ["sample-2"])
        report = orjson.loads(
            (run.cfg.artifacts.dir / "result_cache/report.json").read_bytes()
        )
        assert report["cached_samples"] == (2 if all_cached else 1)
        assert report["pending_samples"] == (0 if all_cached else 1)
    finally:
        await manager._close_cache_manager()
        await manager._backing_store._cleanup()


@pytest.mark.parametrize(
    "resume,streaming", [(False, True), (True, False), (True, True)]
)
async def test_manager_streams_bounded_batches_with_optional_recovery(
    tmp_path: Path,
    resume: bool,
    streaming: bool,
) -> None:
    history = tmp_path / "raw_records_old.jsonl"
    run = make_run(tmp_path, [history] if resume else [])
    dataset = run.cfg.datasets[0]
    dataset.streaming = streaming
    dataset.streaming_batch_size = 2
    dataset.path.write_bytes(
        b"".join(
            orjson.dumps({"session_id": f"sample-{index}", "text": "hello"}) + b"\n"
            for index in range(7)
        )
    )
    samples = list(CustomDatasetComposer(run=run, tokenizer=None).iter_conversations())
    history.write_bytes(
        b"".join(
            orjson.dumps(raw_record(run, sample)) + b"\n" for sample in samples[:4]
        )
    )
    manager = DatasetManager(run=run)
    manager.publish = AsyncMock()
    if resume:
        manager._cache_manager = CacheManager(run)
    writes: list[list[str]] = []
    add_conversations = MemoryMapDatasetBackingStore.add_conversations

    async def write_batch(
        store: MemoryMapDatasetBackingStore, batch: dict[str, Conversation]
    ) -> None:
        writes.append(list(batch))
        await add_conversations(store, batch)

    try:
        with (
            patch.object(
                MemoryMapDatasetBackingStore, "add_conversations", new=write_batch
            ),
            patch.object(
                SingleTurnDatasetLoader,
                "load_dataset",
                side_effect=AssertionError("must stream input"),
            ),
        ):
            await manager._configure_dataset()
        expected = [
            sample.session_id for sample in (samples[4:] if resume else samples)
        ]
        assert [session_id for batch in writes for session_id in batch] == expected
        assert all(0 < len(batch) <= 2 for batch in writes)
        assert [
            sample.conversation_id for sample in manager.dataset_metadata.conversations
        ] == expected
        assert manager.dataset == {}
        if resume:
            assert [len(batch) for batch in writes] == [2, 1]
            assert manager.dataset_metadata.result_cache.cached_samples == 4
            assert manager.dataset_metadata.result_cache.pending_samples == 3
        else:
            assert manager.dataset_metadata.result_cache is None
    finally:
        await manager._close_cache_manager()
        if manager._backing_store is not None:
            await manager._backing_store._cleanup()


@pytest.mark.parametrize(
    "session_ids", [["same", "same"], ["same", "other", "same"], [None]]
)
async def test_shared_stream_rejects_invalid_recovery_identity(
    tmp_path: Path,
    session_ids: list[str | None],
) -> None:
    history = tmp_path / "raw_records_old.jsonl"
    history.touch()
    run = make_run(tmp_path, [history])
    run.cfg.datasets[0].streaming_batch_size = 1
    run.cfg.datasets[0].path.write_bytes(
        b"".join(
            orjson.dumps({"session_id": session_id, "text": "hello"}) + b"\n"
            for session_id in session_ids
        )
    )
    manager = DatasetManager(run=run)
    manager.publish = AsyncMock()
    manager._cache_manager = CacheManager(run)
    try:
        with pytest.raises(ValueError, match="session_id"):
            await manager._configure_dataset()
        manager.publish.assert_not_awaited()
    finally:
        await manager._close_cache_manager()
        if manager._backing_store is not None:
            await manager._backing_store._cleanup()


async def test_ordinary_stream_keeps_grouped_sessions_and_rejects_empty_input(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path, [])
    dataset = run.cfg.datasets[0]
    dataset.streaming = True
    dataset.streaming_batch_size = 1
    dataset.path.write_bytes(
        b'{"session_id":"same","text":"first"}\n{"session_id":"same","text":"second"}\n'
    )
    manager = DatasetManager(run=run)
    manager.publish = AsyncMock()
    try:
        await manager._configure_dataset()
        assert len(manager.dataset_metadata.conversations) == 1
        assert manager.dataset_metadata.total_turn_count == 2
    finally:
        if manager._backing_store is not None:
            await manager._backing_store._cleanup()
    dataset.path.write_bytes(b"")
    manager = DatasetManager(run=run)
    manager.publish = AsyncMock()
    try:
        with pytest.raises(ValueError, match="contains no valid conversations"):
            await manager._configure_dataset()
    finally:
        if manager._backing_store is not None:
            await manager._backing_store._cleanup()
