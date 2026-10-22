"""Disk-backed recovery of successful single-turn requests from RAW exports."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Iterator
from io import BufferedReader
from pathlib import Path
from typing import Any, BinaryIO

import orjson
import zstandard

from aiperf.common.models import Conversation
from aiperf.common.models.dataset_models import CacheReport, CacheSource
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.payload_formatting import format_conversation_payloads


def payload_fingerprint(payload: dict[str, Any]) -> str:
    """Hash every wire-body field, including model and generation parameters."""
    return hashlib.sha256(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()


def _response_choices(response: Any) -> list[dict[str, Any]] | None:
    if not isinstance(response, dict):
        return None
    body = response.get("text", response.get("data"))
    if body in (None, "[DONE]"):
        return []
    if isinstance(body, str):
        try:
            body = orjson.loads(body)
        except orjson.JSONDecodeError:
            return None
    if not isinstance(body, dict) or body.get("error"):
        return None
    choices = body.get("choices") or []
    return choices if isinstance(choices, list) else None


def _choice_result(choice: Any) -> tuple[int, bool, bool] | None:
    if not isinstance(choice, dict):
        return None
    index = choice.get("index", 0)
    message = choice.get("message") or choice.get("delta") or {}
    reason = choice.get("finish_reason")
    if (
        not isinstance(index, int)
        or not isinstance(message, dict)
        or reason not in (None, "stop")
    ):
        return None
    content = message.get("content")
    return index, isinstance(content, str) and bool(content.strip()), reason == "stop"


def _complete_responses(responses: list[Any], expected: int) -> bool:
    nonempty: set[int] = set()
    finished: set[int] = set()
    for response in responses:
        choices = _response_choices(response)
        if choices is None:
            return False
        for choice in choices:
            result = _choice_result(choice)
            if result is None:
                return False
            index, has_content, stopped = result
            if has_content:
                nonempty.add(index)
            if stopped:
                finished.add(index)
    return finished == set(range(expected)) and finished.issubset(nonempty)


def _successful_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    metadata = record.get("metadata") or {}
    payload = record.get("payload")
    if (
        record.get("error")
        or record.get("status") != 200
        or not isinstance(metadata, dict)
        or metadata.get("benchmark_phase") != "profiling"
        or metadata.get("turn_index") != 0
        or metadata.get("was_cancelled")
        or metadata.get("context_overflow_skip")
        or not isinstance(payload, dict)
        or not payload.get("model")
    ):
        return None
    expected = payload.get("n", 1)
    responses = record.get("responses") or []
    if not isinstance(expected, int) or expected < 1 or not isinstance(responses, list):
        return None
    if not _complete_responses(responses, expected):
        return None
    return payload


class CacheManager:
    """Index historical successes on disk and select pending conversations.

    Work runs on the dataset preparation thread. Only IDs, hashes and source
    references live in SQLite; response bodies are never retained in memory.
    """

    def __init__(self, run: BenchmarkRun) -> None:
        self.run = run
        self.report = CacheReport()
        self.directory = run.cfg.artifacts.dir / "result_cache"
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "reused.jsonl").write_bytes(b"")
        self.database = sqlite3.connect(
            self.directory / "index.sqlite3", check_same_thread=False
        )
        self.database.execute("PRAGMA cache_size=-8192")
        self.database.execute(
            "CREATE TABLE IF NOT EXISTS successes (id TEXT, fingerprint TEXT, source TEXT, line INTEGER, PRIMARY KEY(id, fingerprint))"
        )
        self.database.execute("CREATE TABLE IF NOT EXISTS inputs (id TEXT PRIMARY KEY)")
        self.database.execute("DELETE FROM successes")
        self.database.execute("DELETE FROM inputs")
        try:
            self._index_sources()
        except BaseException:
            self.close()
            raise

    def _index_sources(self) -> None:
        dataset = self.run.cfg.get_default_dataset()
        files: set[Path] = set()
        for source in dataset.resume_from:
            source = source.expanduser().resolve(strict=True)
            candidates = (
                [source]
                if source.is_file()
                else [
                    path
                    for path in source.rglob("*")
                    if path.is_file()
                    and (
                        path.name.startswith("raw_records_")
                        or path.name == "profile_export_raw.jsonl"
                        or path.name == "profile_export_raw.jsonl.zst"
                    )
                    and (
                        path.name.endswith(".jsonl") or path.name.endswith(".jsonl.zst")
                    )
                ]
            )
            if not candidates:
                raise ValueError(f"No RAW JSONL files found in resume_from: {source}")
            files.update(candidates)
        current = self.run.cfg.artifacts.dir.resolve()
        for path in sorted(files):
            if (
                path.is_relative_to(current)
                or path == dataset.path.expanduser().resolve()
            ):
                raise ValueError(
                    "resume_from must not include the current artifact directory"
                )
            size_bytes = path.stat().st_size
            self.report.sources.append(
                CacheSource(path=str(path), size_bytes=size_bytes)
            )
            opener = zstandard.open if path.suffix == ".zst" else open
            with opener(path, "rb") as handle, BufferedReader(handle) as buffered:
                self._index_file(path, buffered, size_bytes)
            if path.suffix == ".zst" and path.stat().st_size != size_bytes:
                raise ValueError(
                    f"Compressed recovery source changed during indexing: {path}"
                )
        self.database.commit()
        self.report.successful_records = self.database.execute(
            "SELECT count(*) FROM successes"
        ).fetchone()[0]

    def _index_file(self, path: Path, handle: BinaryIO, size_bytes: int) -> None:
        remaining = size_bytes if path.suffix != ".zst" else None
        line_number = 0
        while remaining is None or remaining > 0:
            line = (
                handle.readline() if remaining is None else handle.readline(remaining)
            )
            if not line:
                if remaining:
                    raise ValueError(
                        f"Recovery source truncated during indexing: {path}"
                    )
                break
            line_number += 1
            if remaining is not None:
                remaining -= len(line)
            if not line.endswith(b"\n"):
                self.report.incomplete_tails += 1
                break
            if not line.strip():
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError as error:
                raise ValueError(f"Invalid RAW JSON at {path}:{line_number}") from error
            self.report.scanned_records += 1
            if not isinstance(record, dict):
                raise ValueError(
                    f"RAW record must be an object at {path}:{line_number}"
                )
            payload = _successful_payload(record)
            metadata = record.get("metadata") or {}
            session_id = (
                metadata.get("conversation_id") if isinstance(metadata, dict) else None
            )
            if payload is None or not isinstance(session_id, str) or not session_id:
                self.report.ignored_records += 1
                continue
            cursor = self.database.execute(
                "INSERT OR IGNORE INTO successes VALUES (?, ?, ?, ?)",
                (session_id, payload_fingerprint(payload), str(path), line_number),
            )
            self.report.duplicate_successes += cursor.rowcount == 0

    def pending(self, conversations: Iterable[Conversation]) -> Iterator[Conversation]:
        """Yield only unmatched samples, rejecting grouped or duplicate IDs."""
        endpoint = ModelEndpointInfo.from_run(self.run)
        with (self.directory / "reused.jsonl").open("ab") as reused:
            for conversation in conversations:
                if len(conversation.turns) != 1:
                    raise ValueError(
                        "resume_from requires exactly one turn per explicit session_id"
                    )
                try:
                    self.database.execute(
                        "INSERT INTO inputs VALUES (?)", (conversation.session_id,)
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        f"Duplicate session_id in recovery input: {conversation.session_id!r}"
                    ) from error
                _, _, payload = next(
                    format_conversation_payloads([conversation], endpoint)
                )
                row = self.database.execute(
                    "SELECT source, line FROM successes WHERE id=? AND fingerprint=?",
                    (conversation.session_id, payload_fingerprint(payload)),
                ).fetchone()
                self.report.total_samples += 1
                if row is not None:
                    self.report.cached_samples += 1
                    reused.write(
                        orjson.dumps(
                            {
                                "conversation_id": conversation.session_id,
                                "source": row[0],
                                "line": row[1],
                            }
                        )
                        + b"\n"
                    )
                else:
                    self.report.pending_samples += 1
                    yield conversation

    def save_report(self) -> None:
        """Persist recovery accounting without changing historical files."""
        self.database.commit()
        temporary = self.directory / "report.json.tmp"
        temporary.write_bytes(self.report.model_dump_json(indent=2).encode())
        temporary.replace(self.directory / "report.json")

    def close(self) -> None:
        """Release the preparation-only SQLite connection."""
        self.database.close()
