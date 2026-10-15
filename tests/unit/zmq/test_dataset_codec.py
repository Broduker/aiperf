"""Dataset notification fidelity, responsiveness, and replay regression tests."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import orjson
import pytest

from aiperf.common.enums import MemoryMapFormat, MessageType
from aiperf.common.messages import DatasetConfiguredNotification, Message
from aiperf.common.messages.dataset_messages import DatasetConfigStatusResponse
from aiperf.common.models.dataset_models import (
    ConversationMetadata,
    DatasetMetadata,
    MemoryMapClientMetadata,
    TurnMetadata,
)
from aiperf.zmq.dataset_codec import decode_message, encode_message
from aiperf.zmq.sub_client import ZMQSubClient
from aiperf.zmq.zmq_defaults import TOPIC_END_ENCODED


def notification(count: int = 1) -> DatasetConfiguredNotification:
    return DatasetConfiguredNotification(
        service_id="dataset_manager",
        dataset_generation="first",
        metadata=DatasetMetadata(
            sampling_strategy="sequential",
            conversations=[
                ConversationMetadata(
                    conversation_id=f"{index:08d}-" + "x" * 512,
                    turns=[TurnMetadata()],
                )
                for index in range(count)
            ],
        ),
        client_metadata=MemoryMapClientMetadata(
            format=MemoryMapFormat.PAYLOAD_BYTES,
            data_file_path=Path("/dataset.dat"),
            index_file_path=Path("/index.dat"),
            conversation_count=count,
            total_size_bytes=1024,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("catch_up", [False, True])
async def test_sparse_configuration_preserves_nondefault_metadata(
    catch_up: bool,
) -> None:
    original = notification()
    original.metadata.conversations[0].is_root = False
    original.metadata.conversations[0].parent_conversation_id = "parent"
    original.metadata.conversations[0].accuracy_ground_truth = "answer"
    original.metadata.conversations[0].turns[0].delay_ms = 123
    original.metadata.conversations[0].turns[0].has_forks = True
    message = (
        DatasetConfigStatusResponse(
            service_id="dataset_manager", request_id="catchup", notification=original
        )
        if catch_up
        else original
    )
    payload = await encode_message(message)
    assert b'"is_orchestrator"' not in payload
    assert Message.from_json(payload) == message
    with patch("aiperf.zmq.dataset_codec._LARGE_MESSAGE_BYTES", 0):
        restored = await decode_message(payload)
    assert restored == message


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_large_configuration_yields_during_encoding_and_validation(
    legacy: bool,
) -> None:
    original = notification(4096)
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        payload = await encode_message(original)
        assert ticks >= 8
        if legacy:
            payload = original.to_json_bytes()
        before = ticks
        restored = await decode_message(payload)
        assert ticks - before >= 8
        assert restored == original
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_rebroadcast_coalesces_decode_and_delivery(mock_zmq_context) -> None:
    client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)
    callback = AsyncMock()
    client._subscribers[MessageType.DATASET_CONFIGURED_NOTIFICATION] = [callback]
    topic = MessageType.DATASET_CONFIGURED_NOTIFICATION.encode() + TOPIC_END_ENCODED
    payload = await encode_message(notification())
    with patch("aiperf.zmq.sub_client.decode_message", wraps=decode_message) as decode:
        await asyncio.gather(
            *(client._handle_message(topic, payload) for _ in range(4))
        )
        assert decode.call_count == 1
        callback.assert_awaited_once()
        changed = orjson.loads(payload)
        changed["dataset_generation"] = "next"
        await client._handle_message(topic, orjson.dumps(changed))
        assert decode.call_count == 2
        assert callback.await_count == 2


@pytest.mark.asyncio
async def test_failed_configuration_delivery_can_be_retried(mock_zmq_context) -> None:
    client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)
    callback = AsyncMock(side_effect=[RuntimeError("retry"), None])
    client._subscribers[MessageType.DATASET_CONFIGURED_NOTIFICATION] = [callback]
    topic = MessageType.DATASET_CONFIGURED_NOTIFICATION.encode() + TOPIC_END_ENCODED
    payload = await encode_message(notification())
    await client._handle_message(topic, payload)
    await client._handle_message(topic, payload)
    await client._handle_message(topic, payload)
    assert callback.await_count == 2


@pytest.mark.asyncio
async def test_catch_up_without_notification_keeps_normal_validation() -> None:
    message = DatasetConfigStatusResponse(service_id="dataset_manager")
    with patch("aiperf.zmq.dataset_codec._LARGE_MESSAGE_BYTES", 0):
        assert await decode_message(await encode_message(message)) == message
        payload = message.model_dump()
        payload["notification"] = None
        assert await decode_message(orjson.dumps(payload)) == message


@pytest.mark.asyncio
async def test_large_configuration_rejects_invalid_turn_metadata() -> None:
    payload = orjson.loads(await encode_message(notification()))
    payload["metadata"]["conversations"][0]["turns"][0]["delay_ms"] = -1
    with (
        patch("aiperf.zmq.dataset_codec._LARGE_MESSAGE_BYTES", 0),
        pytest.raises(ValueError, match="delay_ms"),
    ):
        await decode_message(orjson.dumps(payload))
