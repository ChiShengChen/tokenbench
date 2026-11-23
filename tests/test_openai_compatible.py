import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from time import sleep

import pytest
from typer.testing import CliRunner

from tokenbench.cli import app

cli = CliRunner()


@dataclass(frozen=True)
class Reply:
    status: int
    data: dict[str, object]
    delay: float = 0


@contextmanager
def mock_server(
    *replies: Reply,
) -> Iterator[tuple[str, list[tuple[str, str | None, dict[str, object]]]]]:
    requests: list[tuple[str, str | None, dict[str, object]]] = []
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                index = len(requests)
                requests.append((self.path, self.headers.get("Authorization"), payload))
            reply = replies[index] if index < len(replies) else Reply(418, {})
            if reply.delay:
                sleep(reply.delay)
            encoded = json.dumps(reply.data).encode()
            try:
                self.send_response(reply.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            thread.join()


def completion(*, cost: float | None = None, finish_reason: str = "stop") -> Reply:
    usage: dict[str, object] = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 80},
        "completion_tokens_details": {"reasoning_tokens": 6},
    }
    if cost is not None:
        usage["cost"] = cost
    return Reply(
        200,
        {
            "choices": [{"message": {"content": "exact answer"}, "finish_reason": finish_reason}],
            "usage": usage,
        },
    )


def make_suite(
    path: Path, base_url: str, *, count: int = 1, retries: int = 0, target_options: str = ""
) -> Path:
    suite = path / "suite.yaml"
    suite.write_text(
        f"""schema_version: 1
suite_id: loopback-chat
target:
  adapter: openai_compatible
  provider: local-mock
  model: fixture-chat
  base_url: {base_url}
{target_options}cases: cases.jsonl
max_retries: {retries}
price_snapshot: prices.json
""",
        encoding="utf-8",
    )
    cases = (
        {
            "task_id": f"task-{index}",
            "task_type": "chat",
            "input": {"messages": [{"role": "user", "content": "respond from the local fixture"}]},
            "grader_spec": {"kind": "exact", "expected": "exact answer"},
        }
        for index in range(count)
    )
    (path / "cases.jsonl").write_text("\n".join(map(json.dumps, cases)), encoding="utf-8")
    (path / "prices.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_id": "fixture-rates",
                "prices": [
                    {
                        "provider": "local-mock",
                        "model": "fixture-chat",
                        "input_usd_per_million": "2",
                        "output_usd_per_million": "4",
                        "cached_input_usd_per_million": "0.25",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return suite


def run_baseline(suite: Path, output: Path, *flags: str) -> None:
    result = cli.invoke(app, ["baseline", "--suite", str(suite), "--out", str(output), *flags])
    assert result.exit_code == 0, (result.output, result.exception)


def test_loopback_reports_provider_and_snapshot_cost_without_double_counting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_CHAT_TOKEN", "fixture-secret-for-loopback")
    output = tmp_path / "output"
    with mock_server(completion(cost=0.0009), completion()) as (url, requests):
        opts = (
            "  api_key_env: MOCK_CHAT_TOKEN\n  max_completion_tokens: 30\n  max_cost_usd: '0.001'\n"
        )
        suite = make_suite(tmp_path, url, count=2, target_options=opts)
        run_baseline(suite, output, "--budget-usd", "0.002")
    assert len(requests) == 2
    request_path, header, payload = requests[0]
    assert request_path == "/v1/chat/completions"
    assert header == "Bearer fixture-secret-for-loopback"
    assert payload == {
        "model": "fixture-chat",
        "n": 1,
        "stream": False,
        "max_completion_tokens": 30,
        "messages": [{"role": "user", "content": "respond from the local fixture"}],
    }
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["manifest"]["execution"] == "openai_compatible"
    totals = summary["totals"]
    assert totals["successful_trials"] == 2
    assert (
        totals["known_input_tokens"],
        totals["known_output_tokens"],
        totals["total_tokens"],
    ) == (200, 40, 240)
    assert (totals["known_cached_input_tokens"], totals["known_reasoning_tokens"]) == (160, 12)
    assert totals["usage_sources"] == {"provider": 2}
    assert totals["cost_sources"] == {"provider": 1, "price_snapshot": 1}
    assert Decimal(totals["total_cost_usd"]) == Decimal("0.00104")
    for artifact in output.iterdir():
        content = artifact.read_bytes()
        assert b"fixture-secret-for-loopback" not in content
        assert b"respond from the local fixture" not in content
        assert b"exact answer" not in content


def test_timeout_and_throttle_execute_new_attempts_and_leave_cost_unknown(tmp_path: Path) -> None:
    slow = Reply(200, completion().data, delay=1.2)
    billed_429 = Reply(
        429, {"usage": {"cost_usd": "0.0001"}, "error": {"message": "internal test text"}}
    )
    output = tmp_path / "output"
    with mock_server(slow, billed_429, completion(cost=0.0002)) as (url, requests):
        suite = make_suite(tmp_path, url, retries=2, target_options="  timeout_seconds: 0.4\n")
        run_baseline(suite, output)
    assert len(requests) == 3
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["successful_trials"] == 1
    assert summary["totals"]["retries"] == 2
    assert summary["totals"]["total_cost_usd"] is None
    assert summary["totals"]["total_tokens"] is None
    assert Decimal(summary["totals"]["known_cost_usd"]) == Decimal("0.0003")
    with closing(sqlite3.connect(output / "baseline.sqlite3")) as connection:
        assert connection.execute(
            "SELECT status, error_type FROM model_calls ORDER BY id"
        ).fetchall() == [
            ("error", "RequestTimeout"),
            ("error", "RetryableHTTPError"),
            ("completed", None),
        ]


def test_terminal_failures_are_counted_once_and_preserve_billed_usage(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with mock_server(Reply(401, {}), completion(cost=0.0002, finish_reason="length")) as (
        url,
        requests,
    ):
        run_baseline(make_suite(tmp_path, url, count=2, retries=3), output)
    assert len(requests) == 2
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["trials"] == 2
    assert summary["totals"]["successful_trials"] == summary["totals"]["retries"] == 0
    assert [item["error_type"] for item in summary["attempts"]] == [
        "HTTPRequestRejected",
        "InvalidCompletion",
    ]
    assert Decimal(summary["totals"]["known_cost_usd"]) == Decimal("0.0002")


def test_request_is_blocked_before_sending_when_cost_bound_exceeds_budget(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with mock_server(completion()) as (url, requests):
        suite = make_suite(tmp_path, url, target_options="  max_cost_usd: '0.1'\n")
        run_baseline(suite, output, "--budget-usd", "0.05")
    assert not requests
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "max_usd"
    assert summary["totals"]["model_calls"] == 0


def test_dry_run_validates_messages_and_rejects_opaque_policy_hooks(tmp_path: Path) -> None:
    suite = make_suite(tmp_path, "http://127.0.0.1:1/v1")
    result = cli.invoke(app, ["baseline", "--suite", str(suite), "--dry-run"])
    assert result.exit_code == 0, result.output
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        """schema_version: 1
experiment_id: opaque-target
suite: suite.yaml
candidates:
  - name: topk
    policies: [{id: rag.topk, parameters: {k: 1}}]
""",
        encoding="utf-8",
    )
    result = cli.invoke(app, ["run", "--experiment", str(experiment), "--dry-run"])
    assert result.exit_code == 1 and "requires the retrieval hook" in result.output
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        cases.read_text(encoding="utf-8").replace('"user"', '"tool"'), encoding="utf-8"
    )
    result = cli.invoke(app, ["baseline", "--suite", str(suite), "--dry-run"])
    assert result.exit_code == 1 and "Invalid Chat Completions input" in result.output
