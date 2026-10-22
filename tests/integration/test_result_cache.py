"""Exercise resumed preparation and zero-work completion through real services."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path

import orjson
import pytest
import pytest_asyncio
import yaml

from aiperf.common import random_generator as rng
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.dataset.cache_manager import CacheManager
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.dataset.payload_formatting import format_conversation_payloads
from tests.harness.utils import AIPerfCLI, AIPerfMockServer


@pytest_asyncio.fixture
async def recovery_server(
    mock_server_factory: Callable[..., AbstractAsyncContextManager[AIPerfMockServer]],
) -> AsyncIterator[AIPerfMockServer]:
    async with mock_server_factory(fast=True, workers=1) as server:
        yield server


@pytest.mark.integration
@pytest.mark.parametrize("cached_count", [0, 1, 2])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("continuous", [False, True])
async def test_resumed_run_sends_only_pending_requests(
    cli: AIPerfCLI,
    recovery_server: AIPerfMockServer,
    tmp_path: Path,
    temp_output_dir: Path,
    cached_count: int,
    workers: int,
    continuous: bool,
) -> None:
    source = tmp_path / "input.jsonl"
    source.write_bytes(
        b'{"session_id":"one","text":"hello"}\n{"session_id":"two","text":"hello"}\n'
    )
    benchmark = {
        "models": ["test-model"],
        "tokenizer": {"name": "builtin"},
        "endpoint": {
            "type": "chat",
            "urls": [recovery_server.url],
            "wait_for_model_timeout": 0,
        },
        "datasets": [
            {
                "name": "main",
                "type": "file",
                "format": "single_turn",
                "path": str(source),
                "cache_dir": str(temp_output_dir / "persistent"),
            }
        ],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "requests": 50,
            }
        ],
        "runtime": {"ui": "none", "workers": workers, "record_processors": workers},
        "artifacts": {"raw": True, "auto_plot": False},
    }
    if continuous:
        benchmark["phases"][0].update(
            concurrency=2,
            adaptive_scale=True,
            duration=-1,
            adaptive_sustain_duration=1,
            sla={"request_latency": {"avg": {"le": 10000}}},
        )
    cfg = BenchmarkConfig.model_validate(benchmark)
    run = BenchmarkRun(
        benchmark_id="cache-fixture", cfg=cfg, artifact_dir=temp_output_dir
    )
    history = CacheManager.record_path(run, "history")
    history.parent.mkdir(parents=True, exist_ok=True)
    rng.reset()
    rng.init(42)
    samples = list(CustomDatasetComposer(run=run, tokenizer=None).iter_conversations())
    payloads = list(
        format_conversation_payloads(samples, ModelEndpointInfo.from_run(run))
    )
    history.write_bytes(
        b"".join(
            orjson.dumps(
                {
                    "metadata": {
                        "session_num": 0,
                        "request_start_ns": 1,
                        "request_end_ns": 2,
                        "worker_id": "historical-worker",
                        "record_processor_id": "historical-processor",
                        "conversation_id": session_id,
                        "turn_index": 0,
                        "benchmark_phase": "profiling",
                    },
                    "status": 200,
                    "payload": payload,
                    "responses": [
                        {
                            "perf_ns": 2,
                            "text": orjson.dumps(
                                {
                                    "choices": [
                                        {
                                            "index": 0,
                                            "message": {"content": "historical answer"},
                                            "finish_reason": "stop",
                                        }
                                    ]
                                }
                            ).decode(),
                        }
                    ],
                }
            )
            + b"\n"
            for session_id, _, payload in payloads[:cached_count]
        )
    )
    config = tmp_path / "resume.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": benchmark,
            }
        )
    )
    result = await cli.run(
        f'aiperf profile --config "{config}" --tokenizer builtin', timeout=90
    )
    assert result.exit_code == 0
    assert result.request_count == 2 - cached_count
    report = orjson.loads((temp_output_dir / "result_cache/report.json").read_bytes())
    assert report["cached_samples"] == cached_count
    assert report["pending_samples"] == 2 - cached_count
    results_file = temp_output_dir / "profile_export_raw.jsonl"
    results = [orjson.loads(line) for line in results_file.read_bytes().splitlines()]
    assert {record["metadata"]["conversation_id"] for record in results} == {
        "one",
        "two",
    }
    assert len(results) == 2
    original = results_file.read_bytes()
    assert history.exists()
    assert not (temp_output_dir / "persistent/results.jsonl").exists()
    assert not (temp_output_dir / "raw_records").exists()
    benchmark["phases"][0]["concurrency"] = 2
    config.write_text(yaml.safe_dump({"benchmark": benchmark}))
    resumed = await cli.run(
        f'aiperf profile --config "{config}" --tokenizer builtin', timeout=90
    )
    assert resumed.exit_code == 0
    assert resumed.request_count == 0
    assert results_file.read_bytes() == original
