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
@pytest.mark.parametrize(
    "resume,all_cached", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("continuous", [False, True])
async def test_streaming_run_with_optional_recovery(
    cli: AIPerfCLI,
    recovery_server: AIPerfMockServer,
    tmp_path: Path,
    temp_output_dir: Path,
    all_cached: bool,
    resume: bool,
    continuous: bool,
) -> None:
    source = tmp_path / "input.jsonl"
    source.write_bytes(
        b'{"session_id":"one","text":"hello"}\n{"session_id":"two","text":"hello"}\n'
    )
    history = tmp_path / "raw_records_history.jsonl"
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
                "resume_from": [str(history)] if resume else [],
                "streaming": not resume,
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
        "runtime": {"ui": "none", "workers": 1},
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
        benchmark["datasets"][0]["streaming"] = True
    cfg = BenchmarkConfig.model_validate(benchmark)
    run = BenchmarkRun(
        benchmark_id="cache-fixture", cfg=cfg, artifact_dir=temp_output_dir
    )
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
                        "conversation_id": session_id,
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
                                            "message": {"content": "historical answer"},
                                            "finish_reason": "stop",
                                        }
                                    ]
                                }
                            ).decode()
                        }
                    ],
                }
            )
            + b"\n"
            for session_id, _, payload in (payloads if all_cached else payloads[:1])
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
    if resume:
        assert result.request_count == (0 if all_cached else 1)
        report = orjson.loads(
            (temp_output_dir / "result_cache/report.json").read_bytes()
        )
        assert report["cached_samples"] == (2 if all_cached else 1)
        assert report["pending_samples"] == (0 if all_cached else 1)
    else:
        assert result.request_count == (2 if continuous else 50)
        assert not (temp_output_dir / "result_cache").exists()
