"""Cooperative JSON encoding for dataset configuration on both bus channels."""

import asyncio
from typing import Any

import msgspec
import orjson
from pydantic import TypeAdapter

from aiperf.common.enums import MessageType
from aiperf.common.messages import Message
from aiperf.common.messages.dataset_messages import (
    DatasetConfigStatusResponse,
    DatasetConfiguredNotification,
)
from aiperf.common.models.dataset_models import ConversationMetadata, DatasetMetadata

_BATCH_SIZE = 512
_LARGE_MESSAGE_BYTES = 1024 * 1024
_CONVERSATIONS_ADAPTER = TypeAdapter(list[ConversationMetadata])


async def _encode_notification(message: DatasetConfiguredNotification) -> bytes:
    header = message.model_dump(
        exclude={"metadata"},
        exclude_none=True,
        mode="json",
        context={"include_internal": True},
    )
    metadata = message.metadata.model_dump(
        exclude={"conversations"},
        exclude_defaults=True,
        mode="json",
    )
    chunks: list[bytes] = []
    conversations = message.metadata.conversations
    for start in range(0, len(conversations), _BATCH_SIZE):
        chunks.append(
            orjson.dumps(
                [
                    conversation.model_dump(exclude_defaults=True, mode="json")
                    for conversation in conversations[start : start + _BATCH_SIZE]
                ]
            )[1:-1]
        )
        await asyncio.sleep(0)
    metadata_bytes = (
        orjson.dumps(metadata)[:-1] + b',"conversations":[' + b",".join(chunks) + b"]}"
    )
    return orjson.dumps(header)[:-1] + b',"metadata":' + metadata_bytes + b"}"


async def encode_message(message: Message) -> bytes:
    """Yield between metadata batches without changing the JSON message schema."""
    if isinstance(message, DatasetConfiguredNotification):
        return await _encode_notification(message)
    if (
        isinstance(message, DatasetConfigStatusResponse)
        and message.notification is not None
    ):
        header = message.model_dump(
            exclude={"notification"},
            exclude_none=True,
            mode="json",
            context={"include_internal": True},
        )
        notification = await _encode_notification(message.notification)
        return orjson.dumps(header)[:-1] + b',"notification":' + notification + b"}"
    return message.to_json_bytes()


async def _decode_notification(payload: bytes | msgspec.Raw) -> dict[str, Any]:
    fields = msgspec.json.decode(payload, type=dict[str, msgspec.Raw])
    metadata_fields = msgspec.json.decode(
        fields.pop("metadata"), type=dict[str, msgspec.Raw]
    )
    raw_conversations = msgspec.json.decode(
        metadata_fields.pop("conversations", msgspec.Raw(b"[]")),
        type=list[msgspec.Raw],
    )
    conversations: list[ConversationMetadata] = []
    for start in range(0, len(raw_conversations), _BATCH_SIZE):
        end = start + _BATCH_SIZE
        conversations.extend(
            _CONVERSATIONS_ADAPTER.validate_json(
                b"[" + b",".join(raw_conversations[start:end]) + b"]"
            )
        )
        await asyncio.sleep(0)
    metadata = orjson.loads(msgspec.json.encode(metadata_fields))
    metadata["conversations"] = conversations
    notification = orjson.loads(msgspec.json.encode(fields))
    notification["metadata"] = DatasetMetadata.model_validate(metadata)
    return notification


async def decode_message(payload: bytes) -> Message:
    """Validate large dataset notifications in batches so heartbeats can run."""
    if len(payload) < _LARGE_MESSAGE_BYTES:
        return Message.from_json(payload)
    fields = msgspec.json.decode(payload, type=dict[str, msgspec.Raw])
    message_type = orjson.loads(memoryview(fields["message_type"]))
    if message_type == MessageType.DATASET_CONFIGURED_NOTIFICATION:
        data = await _decode_notification(payload)
    elif message_type == MessageType.DATASET_CONFIG_STATUS_RESPONSE and fields.get(
        "notification"
    ) not in (None, msgspec.Raw(b"null")):
        notification = await _decode_notification(fields.pop("notification"))
        data = orjson.loads(msgspec.json.encode(fields))
        data["notification"] = notification
    else:
        return Message.from_json(payload)
    return Message.from_json(data)
