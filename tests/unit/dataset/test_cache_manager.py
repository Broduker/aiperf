"""Recoverable RAW shards use stable payload identity and bounded disk indexing."""

import os
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import orjson
import pytest
import zstandard
from filelock import Timeout

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
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.post_processors.raw_record_writer_processor import (
    RawRecordAggregator,
    RawRecordWriterProcessor,
)


def make_run(tmp_path: Path) -> BenchmarkRun:
    run = _default_test_run()
    source = tmp_path / "input.jsonl"
    source.touch()
    run.cfg.datasets = [
        FileDataset(
            name="default",
            type="file",
            format="single_turn",
            path=source,
            cache_dir=tmp_path / "cache",
        )
    ]
    run.cfg.artifacts.dir = tmp_path / "run"
    run.cfg.artifacts.raw = True
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


def write_shard(run: BenchmarkRun, records: list[dict]) -> Path:
    path = CacheManager.record_path(run, "historical")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(orjson.dumps(record) + b"\n" for record in records))
    return path


def test_recovery_deduplicates_success_without_copying_response_bodies(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path)
    good = raw_record(run, conversation())
    history = write_shard(run, [good, {**good, "error": {"message": "failed"}}, good])
    original = history.read_bytes()
    cache = CacheManager(run)
    try:
        assert [
            sample.session_id
            for sample in cache.pending([conversation(), conversation("second")])
        ] == ["second"]
        cache.save_report()
        assert cache.report.successful_records == 1
        assert cache.report.cached_samples == 1
        assert cache.report.pending_samples == 1
        assert cache.report.duplicate_successes == 1
        assert cache.report.ignored_records == 1
        reused = orjson.loads((cache.directory / "reused.jsonl").read_bytes())
        assert reused["source"] == str(history)
        assert history.read_bytes() == original
        assert not (run.cfg.datasets[0].cache_dir / "results.jsonl").exists()
    finally:
        cache.close()


def test_pending_close_releases_index_read_snapshot(tmp_path: Path) -> None:
    cache = CacheManager(make_run(tmp_path))
    pending = cache.pending([conversation(), conversation("second")])
    try:
        assert next(pending).session_id == "sample-1"
        assert cache.index.in_transaction
        with sqlite3.connect(cache.cache_dir / "index.sqlite3", timeout=0) as writer:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer.execute("BEGIN EXCLUSIVE")
            pending.close()
            assert not cache.index.in_transaction
            writer.execute("BEGIN EXCLUSIVE")
    finally:
        pending.close()
        cache.close()


@pytest.mark.parametrize("fail", [False, True])
def test_pending_nested_call_preserves_outer_read_transaction(
    tmp_path: Path, fail: bool
) -> None:
    cache = CacheManager(make_run(tmp_path))
    pending = cache.pending([conversation()])
    try:
        assert next(pending).session_id == "sample-1"
        if fail:
            with pytest.raises(ValueError, match="Duplicate session_id"):
                list(cache.pending([conversation()]))
        else:
            assert [
                sample.session_id for sample in cache.pending([conversation("second")])
            ] == ["second"]
        assert cache.index.in_transaction
        pending.close()
        assert not cache.index.in_transaction
    finally:
        pending.close()
        cache.close()


@pytest.mark.parametrize("all_cached", [False, True])
def test_pending_exhaustion_allows_refresh_and_export(
    tmp_path: Path, all_cached: bool
) -> None:
    run = make_run(tmp_path)
    first = raw_record(run, conversation())
    second = raw_record(run, conversation("second"))
    write_shard(run, [first, second] if all_cached else [first])
    cache = CacheManager(run)
    try:
        assert [
            sample.session_id
            for sample in cache.pending([conversation(), conversation("second")])
        ] == ([] if all_cached else ["second"])
        assert not cache.index.in_transaction
        cache.save_report()
        if not all_cached:
            write_shard(run, [second])
        cache.refresh()
        output = tmp_path / "export.jsonl"
        assert cache.export(output) == 2
        assert [orjson.loads(line) for line in output.read_bytes().splitlines()] == [
            first,
            second,
        ]
        assert not cache.index.in_transaction
    finally:
        cache.close()


@pytest.mark.parametrize("mutation", ["prompt", "model", "max_tokens", "extra"])
def test_changed_request_is_never_reused(tmp_path: Path, mutation: str) -> None:
    run = make_run(tmp_path)
    record = raw_record(run, conversation())
    if mutation == "prompt":
        record["payload"]["messages"] = [{"role": "user", "content": "other"}]
    else:
        record["payload"][mutation] = "changed"
    write_shard(run, [record])
    cache = CacheManager(run)
    try:
        assert list(cache.pending([conversation()]))
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
    run = make_run(tmp_path)
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
    record = raw_record(make_run(tmp_path), conversation())
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


def test_incremental_index_restarts_without_reading_unchanged_shards(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path)
    history = write_shard(run, [raw_record(run, conversation())])
    cache = CacheManager(run)
    cache.close()
    cache = CacheManager(run)
    try:
        assert cache.report.scanned_records == 0
        assert not list(cache.pending([conversation()]))
    finally:
        cache.close()
    with history.open("ab") as output:
        output.write(orjson.dumps(raw_record(run, conversation("second"))) + b"\n")
    cache = CacheManager(run)
    try:
        assert cache.report.scanned_records == 1
        assert not list(cache.pending([conversation(), conversation("second")]))
    finally:
        cache.close()


def test_index_rebuild_and_unterminated_tail_do_not_modify_shards(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path)
    history = write_shard(run, [raw_record(run, conversation())])
    with history.open("ab") as output:
        output.write(b'{"partial":')
    original = history.read_bytes()
    cache = CacheManager(run)
    assert cache.report.incomplete_tails == 1
    cache.close()
    (run.cfg.datasets[0].cache_dir / "index.sqlite3").unlink()
    cache = CacheManager(run)
    try:
        assert not list(cache.pending([conversation()]))
        assert history.read_bytes() == original
    finally:
        cache.close()


@pytest.mark.parametrize("change", ["missing", "truncated", "replaced"])
def test_missing_or_modified_shards_fail_closed(tmp_path: Path, change: str) -> None:
    run = make_run(tmp_path)
    history = write_shard(run, [raw_record(run, conversation())])
    CacheManager(run).close()
    if change == "missing":
        history.unlink()
    elif change == "truncated":
        history.write_bytes(b"")
    else:
        replacement = history.with_suffix(".replacement")
        replacement.write_bytes(history.read_bytes())
        replacement.replace(history)
    with pytest.raises(ValueError, match="missing|truncated|replaced"):
        CacheManager(run)


def test_invalid_complete_record_fails_and_active_jobs_are_exclusive(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path)
    cache = CacheManager(run)
    try:
        with pytest.raises(Timeout):
            CacheManager(run)
    finally:
        cache.close()
    history = write_shard(run, [])
    history.write_bytes(b"{broken}\n")
    with pytest.raises(ValueError, match="Invalid RAW JSON"):
        CacheManager(run)


def test_duplicate_input_and_missing_explicit_id_are_rejected(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    cache = CacheManager(run)
    try:
        with pytest.raises(ValueError, match="Duplicate session_id"):
            list(cache.pending([conversation(), conversation()]))
        assert not cache.index.in_transaction
    finally:
        cache.close()
    run.cfg.datasets[0].path.write_bytes(b'{"text":"hello"}\n')
    with pytest.raises(ValueError, match="explicit non-empty session_id"):
        SingleTurnDatasetLoader(
            run=run, filename=run.cfg.datasets[0].path
        ).load_dataset()


def test_removed_resume_from_and_unsupported_cache_configs_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        FileDataset(
            name="default", type="file", path=tmp_path / "input", resume_from=[tmp_path]
        )
    with pytest.raises(ValueError, match="path-backed single_turn"):
        FileDataset(
            name="default",
            type="file",
            records=[{"text": "hello"}],
            cache_dir=tmp_path / "cache",
        )
    with pytest.raises(ValueError, match="absolute path"):
        FileDataset(
            name="default",
            type="file",
            path=tmp_path / "input",
            cache_dir=Path("relative"),
        )
    run = make_run(tmp_path)
    run.cfg.artifacts.raw = False
    with pytest.raises(ValueError, match="artifacts.raw"):
        run.cfg.validate_result_cache_resume()
    run.cfg.artifacts.raw = True
    run.cfg.phases.append(run.cfg.phases[0].model_copy(update={"name": "second"}))
    with pytest.raises(ValueError, match="one dataset and one profiling phase"):
        run.cfg.validate_result_cache_resume()


async def test_multiple_processors_write_only_independent_persistent_shards(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path)
    first = RawRecordWriterProcessor(service_id="processor", run=run)
    second = RawRecordWriterProcessor(service_id="processor", run=run)
    assert first.output_file != second.output_file
    assert first.output_file.is_relative_to(run.cfg.datasets[0].cache_dir / "records")
    await first._open_file()
    await second._open_file()
    good = orjson.dumps(raw_record(run, conversation()))
    other = orjson.dumps(raw_record(run, conversation("second")))
    try:
        with patch(
            "aiperf.post_processors.raw_record_writer_processor.os.fsync",
            wraps=os.fsync,
        ) as fsync:
            await first._flush_buffer([good])
            await second._flush_buffer([good, other])
            assert fsync.call_count == 2
        assert first.output_file.read_bytes() == good + b"\n"
        assert second.output_file.read_bytes() == good + b"\n" + other + b"\n"
        assert not (run.cfg.artifacts.dir / "raw_records").exists()
        cache = CacheManager(run)
        try:
            assert not list(cache.pending([conversation(), conversation("second")]))
        finally:
            cache.close()
    finally:
        await first._close_file()
        await second._close_file()


async def test_shard_fsync_failure_is_retried_before_finalization(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path)
    writer = RawRecordWriterProcessor(service_id="processor", run=run)
    await writer._open_file()
    good = orjson.dumps(raw_record(run, conversation()))
    try:
        with (
            patch(
                "aiperf.post_processors.raw_record_writer_processor.os.fsync",
                side_effect=OSError("disk unavailable"),
            ),
            pytest.raises(OSError, match="disk unavailable"),
        ):
            await writer._flush_buffer([good])
        assert writer._buffer == [good]
        await writer.flush_buffer()
        assert writer._write_error is None
    finally:
        await writer._close_file()


async def test_final_export_joins_all_runs_filters_payload_and_reports_missing(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path)
    first = raw_record(run, conversation())
    write_shard(run, [first, raw_record(run, conversation("outside"))])
    cache = CacheManager(run)
    try:
        assert [
            sample.session_id
            for sample in cache.pending(
                [conversation(), conversation("second"), conversation("missing")]
            )
        ] == ["second", "missing"]
        cache.save_report()
        second = raw_record(run, conversation("second"))
        write_shard(run, [second, first, raw_record(run, conversation(text="changed"))])
        exporter = RawRecordAggregator(
            ExporterConfig(results=None, cfg=run.cfg, telemetry_results=None, run=run)
        )
        await exporter.export()
        results = [
            orjson.loads(line)
            for line in exporter.output_file.read_bytes().splitlines()
        ]
        assert results == [first, second]
        report = orjson.loads((cache.directory / "export_report.json").read_bytes())
        assert report["missing_samples"] == 1
        assert report["complete"] is False
        assert (
            len(list((run.cfg.datasets[0].cache_dir / "records").rglob("*.jsonl"))) == 2
        )
    finally:
        cache.close()


@pytest.mark.parametrize("all_cached", [True, False])
@pytest.mark.parametrize("compressed", [False, True])
async def test_manager_writes_only_pending_and_handles_empty_mmap(
    tmp_path: Path, all_cached: bool, compressed: bool
) -> None:
    run = make_run(tmp_path)
    run.cfg.datasets[0].path.write_bytes(
        b'{"session_id":"sample-1","text":"hello"}\n'
        b'{"session_id":"sample-2","text":"hello"}\n'
    )
    samples = list(CustomDatasetComposer(run=run, tokenizer=None).iter_conversations())
    write_shard(
        run,
        [
            raw_record(run, sample)
            for sample in (samples if all_cached else samples[:1])
        ],
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
    run = make_run(tmp_path)
    dataset = run.cfg.datasets[0]
    if not resume:
        dataset.cache_dir = None
    dataset.streaming = streaming
    dataset.streaming_batch_size = 2
    dataset.path.write_bytes(
        b"".join(
            orjson.dumps({"session_id": f"sample-{index}", "text": "hello"}) + b"\n"
            for index in range(7)
        )
    )
    samples = list(CustomDatasetComposer(run=run, tokenizer=None).iter_conversations())
    if resume:
        write_shard(run, [raw_record(run, sample) for sample in samples[:4]])
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
    run = make_run(tmp_path)
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
    run = make_run(tmp_path)
    dataset = run.cfg.datasets[0]
    dataset.cache_dir = None
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
