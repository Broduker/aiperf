"""Disk-backed recovery of successful single-turn requests from RAW exports."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

import orjson
from filelock import FileLock

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


def _record_identity(record: Any) -> tuple[str, str] | None:
    if not isinstance(record, dict):
        return None
    payload = _successful_payload(record)
    if payload is None:
        return None
    session_id = record["metadata"].get("conversation_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    return session_id, payload_fingerprint(payload)


class CacheManager:
    """Recover durable RAW shards using a disk-only success index.

    Only preparation and final export update the index. Record processors write
    independent append-only shards; they never write SQLite or another copy of
    the response body. All synchronous methods run on service I/O threads.
    """

    def __init__(self, run: BenchmarkRun, *, prepare: bool = True) -> None:
        self.run = run
        cache_dir = run.cfg.get_default_dataset().cache_dir
        if cache_dir is None:
            raise ValueError("CacheManager requires cache_dir")
        self.cache_dir = cache_dir.expanduser().resolve()
        self.records_dir = self.cache_dir / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.directory = run.cfg.artifacts.dir / "result_cache"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.report = CacheReport()
        self._lease = FileLock(self.cache_dir / "active.lock", thread_local=False)
        self.database: sqlite3.Connection | None = None
        self.index: sqlite3.Connection | None = None
        try:
            if prepare:
                self._lease.acquire(timeout=0)
            self.database = sqlite3.connect(
                self.directory / "index.sqlite3", check_same_thread=False
            )
            self.database.execute("PRAGMA cache_size=-8192")
            if prepare:
                self.database.execute("DROP TABLE IF EXISTS inputs")
                self.database.execute(
                    "CREATE TABLE inputs (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, ordinal INTEGER UNIQUE)"
                )
                self.database.commit()
                (self.directory / "reused.jsonl").write_bytes(b"")
            self.index = sqlite3.connect(
                self.cache_dir / "index.sqlite3", check_same_thread=False
            )
            self.index.execute("PRAGMA cache_size=-8192")
            self.index.execute(
                "CREATE TABLE IF NOT EXISTS successes (id TEXT, fingerprint TEXT, source TEXT, "
                "offset INTEGER, length INTEGER, line INTEGER, PRIMARY KEY(id, fingerprint))"
            )
            self.index.execute(
                "CREATE TABLE IF NOT EXISTS shards (source TEXT PRIMARY KEY, offset INTEGER, "
                "line INTEGER, device INTEGER, inode INTEGER)"
            )
            self.index.commit()
            self.refresh()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def record_path(run: BenchmarkRun, service_id: str) -> Path:
        """Allocate a unique durable shard, including across processor restarts."""
        directory = run.cfg.get_default_dataset().cache_dir
        if directory is None:
            raise ValueError("Durable RAW records require cache_dir")
        generation = hashlib.sha256(run.benchmark_id.encode()).hexdigest()[:16]
        safe_id = service_id.replace("/", "_").replace(":", "_").replace(" ", "_")
        return (
            directory
            / "records"
            / generation
            / f"raw_records_{safe_id}_{uuid4().hex}.jsonl"
        )

    def refresh(self) -> None:
        """Index only new complete lines, leaving interrupted tails untouched."""
        assert self.index is not None
        with FileLock(self.cache_dir / "index.lock", timeout=60):
            paths = sorted(self.records_dir.rglob("*.jsonl"))
            present = {str(path.relative_to(self.cache_dir)) for path in paths}
            known = {row[0] for row in self.index.execute("SELECT source FROM shards")}
            if missing := known - present:
                raise ValueError(
                    f"Durable result shards are missing: {sorted(missing)}"
                )
            for path in paths:
                self._index_shard(path)
            self.index.commit()
            self.report.successful_records = self.index.execute(
                "SELECT count(*) FROM successes"
            ).fetchone()[0]

    def _index_shard(self, path: Path) -> None:
        assert self.index is not None
        source = str(path.relative_to(self.cache_dir))
        snapshot = path.stat()
        if path.resolve() != path.absolute() or path.is_symlink():
            raise ValueError(f"Durable result shard must not be a symlink: {path}")
        if (
            path.resolve()
            == self.run.cfg.get_default_dataset().path.expanduser().resolve()
        ):
            raise ValueError("The input dataset must not be stored among result shards")
        row = self.index.execute(
            "SELECT offset, line, device, inode FROM shards WHERE source=?", (source,)
        ).fetchone()
        if snapshot.st_size == 0 and row is None:
            return
        offset, line_number = row[:2] if row else (0, 0)
        if row and (snapshot.st_size < offset or snapshot.st_ino != row[3]):
            raise ValueError(f"Durable result shard was truncated or replaced: {path}")
        self.report.sources.append(
            CacheSource(path=str(path), size_bytes=snapshot.st_size)
        )
        with path.open("rb") as handle:
            handle.seek(offset)
            while offset < snapshot.st_size:
                line = handle.readline(snapshot.st_size - offset)
                if not line or not line.endswith(b"\n"):
                    self.report.incomplete_tails += 1
                    break
                line_number += 1
                self._index_line(source, offset, line_number, line)
                offset += len(line)
        self.index.execute(
            "INSERT OR REPLACE INTO shards VALUES (?, ?, ?, ?, ?)",
            (source, offset, line_number, snapshot.st_dev, snapshot.st_ino),
        )

    def _index_line(
        self, source: str, offset: int, line_number: int, line: bytes
    ) -> None:
        assert self.index is not None
        if not line.strip():
            return
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError as error:
            raise ValueError(f"Invalid RAW JSON at {source}:{line_number}") from error
        self.report.scanned_records += 1
        if not isinstance(record, dict):
            raise ValueError(f"RAW record must be an object at {source}:{line_number}")
        identity = _record_identity(record)
        if identity is None:
            self.report.ignored_records += 1
            return
        cursor = self.index.execute(
            "INSERT OR IGNORE INTO successes VALUES (?, ?, ?, ?, ?, ?)",
            (*identity, source, offset, len(line), line_number),
        )
        self.report.duplicate_successes += cursor.rowcount == 0

    def pending(self, conversations: Iterable[Conversation]) -> Iterator[Conversation]:
        """Yield current samples without a matching durable successful request."""
        assert self.database is not None and self.index is not None
        endpoint = ModelEndpointInfo.from_run(self.run)
        owns_transaction = not self.index.in_transaction
        if owns_transaction:
            self.index.execute("BEGIN")
        try:
            with (self.directory / "reused.jsonl").open("ab") as reused:
                for conversation in conversations:
                    if len(conversation.turns) != 1:
                        raise ValueError(
                            "cache_dir requires exactly one turn per explicit session_id"
                        )
                    _, _, payload = next(
                        format_conversation_payloads([conversation], endpoint)
                    )
                    fingerprint = payload_fingerprint(payload)
                    try:
                        self.database.execute(
                            "INSERT INTO inputs VALUES (?, ?, ?)",
                            (
                                conversation.session_id,
                                fingerprint,
                                self.report.total_samples,
                            ),
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError(
                            f"Duplicate session_id in recovery input: {conversation.session_id!r}"
                        ) from error
                    row = self.index.execute(
                        "SELECT source, line FROM successes WHERE id=? AND fingerprint=?",
                        (conversation.session_id, fingerprint),
                    ).fetchone()
                    self.report.total_samples += 1
                    if row is None:
                        self.report.pending_samples += 1
                        yield conversation
                    else:
                        self.report.cached_samples += 1
                        reused.write(
                            orjson.dumps(
                                {
                                    "conversation_id": conversation.session_id,
                                    "source": str(self.cache_dir / row[0]),
                                    "line": row[1],
                                }
                            )
                            + b"\n"
                        )
        finally:
            if owns_transaction and self.index is not None:
                self.index.rollback()

    def save_report(self) -> None:
        """Persist preparation accounting and the current input selection."""
        assert self.database is not None
        self.database.commit()
        temporary = self.directory / "report.json.tmp"
        temporary.write_bytes(self.report.model_dump_json(indent=2).encode())
        temporary.replace(self.directory / "report.json")

    def export(self, output: Path) -> int:
        """Atomically export one success per current input, in input order.

        Shards remain the authoritative recovery data. Exporting creates a final
        delivery file only after execution, rather than duplicating every write.
        Missing current samples are explicitly counted in the export report.
        """
        assert self.database is not None and self.index is not None
        self.index.execute(
            "ATTACH DATABASE ? AS current_run", (str(self.directory / "index.sqlite3"),)
        )
        temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        exported = 0
        missing = 0
        handles: OrderedDict[str, BinaryIO] = OrderedDict()
        query: sqlite3.Cursor | None = None
        try:
            query = self.index.execute(
                "SELECT inputs.id, inputs.fingerprint, successes.source, successes.offset, successes.length "
                "FROM current_run.inputs LEFT JOIN successes "
                "ON inputs.id=successes.id AND inputs.fingerprint=successes.fingerprint ORDER BY inputs.ordinal"
            )
            with temporary.open("wb") as target:
                for session_id, fingerprint, source, offset, length in query:
                    if source is None:
                        missing += 1
                        continue
                    shard = handles.pop(source, None)
                    if shard is None:
                        if len(handles) >= 32:
                            handles.popitem(last=False)[1].close()
                        shard = (self.cache_dir / source).open("rb")
                    handles[source] = shard
                    shard.seek(offset)
                    line = shard.read(length)
                    if len(line) != length or _record_identity(orjson.loads(line)) != (
                        session_id,
                        fingerprint,
                    ):
                        raise ValueError(
                            f"Durable result changed before export: {source}:{offset}"
                        )
                    target.write(line)
                    exported += 1
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(output)
            report = self.directory / "export_report.json"
            report.write_bytes(
                orjson.dumps(
                    {
                        "total_samples": exported + missing,
                        "successful_samples": exported,
                        "missing_samples": missing,
                        "complete": missing == 0,
                        "output": str(output),
                    },
                    option=orjson.OPT_INDENT_2,
                )
            )
            return exported
        finally:
            if query is not None:
                query.close()
            for shard in handles.values():
                shard.close()
            temporary.unlink(missing_ok=True)
            self.index.execute("DETACH DATABASE current_run")

    def close(self) -> None:
        """Release preparation/export connections and the active-run lease."""
        if self.database is not None:
            self.database.close()
            self.database = None
        if self.index is not None:
            self.index.close()
            self.index = None
        self._lease.release()
