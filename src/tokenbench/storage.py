"""Append attempts and instrumented model calls to a local SQLite ledger."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Literal, TypeAlias

from tokenbench.pricing import UsageRecord

AttemptStatus: TypeAlias = Literal["passed", "failed", "error", "interrupted"]
CallStatus: TypeAlias = Literal["completed", "error", "interrupted"]

_SCHEMA = """
CREATE TABLE run_manifest (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    manifest_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE attempts (
    id INTEGER PRIMARY KEY,
    task_id TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    passed INTEGER,
    error_type TEXT,
    started_at TEXT NOT NULL,
    duration_ms REAL,
    UNIQUE(task_id, repetition, attempt)
);
CREATE TABLE model_calls (
    id INTEGER PRIMARY KEY,
    attempt_id INTEGER NOT NULL REFERENCES attempts(id),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    error_type TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_input_tokens INTEGER,
    reasoning_tokens INTEGER,
    usage_source TEXT NOT NULL DEFAULT 'unknown',
    cost_usd TEXT,
    cost_source TEXT NOT NULL DEFAULT 'unknown',
    started_at TEXT NOT NULL,
    duration_ms REAL
);
CREATE INDEX model_calls_attempt_id ON model_calls(attempt_id);
"""


def now() -> str:
    return datetime.now(UTC).isoformat()


def elapsed_ms(started: int) -> float:
    return (perf_counter_ns() - started) / 1_000_000


class Ledger:
    def __init__(self, path: Path, manifest_json: str) -> None:
        path.touch(exist_ok=False)
        self.connection = sqlite3.connect(path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.executescript(_SCHEMA)
            self.connection.execute(
                "INSERT INTO run_manifest (id, manifest_json, started_at) VALUES (1, ?, ?)",
                (manifest_json, now()),
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def start_attempt(self, task_id: str, repetition: int, attempt: int) -> int:
        cursor = self.connection.execute(
            "INSERT INTO attempts (task_id, repetition, attempt, started_at) VALUES (?, ?, ?, ?)",
            (task_id, repetition, attempt, now()),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def finish_attempt(
        self,
        attempt_id: int,
        status: AttemptStatus,
        started: int,
        error_type: str | None = None,
    ) -> None:
        passed = status == "passed" if status in ("passed", "failed") else None
        self.connection.execute(
            "UPDATE attempts SET status = ?, passed = ?, error_type = ?, duration_ms = ? WHERE id = ?",
            (status, passed, error_type, elapsed_ms(started), attempt_id),
        )

    def start_call(self, attempt_id: int, provider: str, model: str) -> int:
        cursor = self.connection.execute(
            "INSERT INTO model_calls (attempt_id, provider, model, started_at) VALUES (?, ?, ?, ?)",
            (attempt_id, provider, model, now()),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def finish_call(
        self,
        call_id: int,
        status: CallStatus,
        started: int,
        usage: UsageRecord,
        error_type: str | None,
    ) -> None:
        self.connection.execute(
            """UPDATE model_calls SET status = :status, error_type = :error_type,
               duration_ms = :duration_ms, usage_source = :usage_source, cost_source = :cost_source,
               input_tokens = :input_tokens, output_tokens = :output_tokens,
               cached_input_tokens = :cached_input_tokens, reasoning_tokens = :reasoning_tokens,
               cost_usd = :cost_usd WHERE id = :call_id""",
            {
                **usage.model_dump(mode="json"),
                "call_id": call_id,
                "status": status,
                "duration_ms": elapsed_ms(started),
                "error_type": error_type,
            },
        )

    def finish_run(self) -> None:
        self.connection.execute("UPDATE run_manifest SET finished_at = ? WHERE id = 1", (now(),))
