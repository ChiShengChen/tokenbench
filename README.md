# TokenBench

TokenBench runs local baselines for Python applications. It executes each case against a callable, grades its JSON result by exact match, and records attempts, model calls, and usage in SQLite. The bundled toy RAG suite runs end to end against frozen records and a deterministic fixture model.

## Install and run the offline demo

TokenBench requires Python 3.11 or later. Install the dependencies with `uv sync` from the project root. The initial setup fetches the locked dependencies. Running the demo requires zero API keys and zero network access.

```sh
uv sync
uv run tokenbench doctor
uv run tokenbench init --path ./tokenbench-demo
uv run tokenbench baseline --suite ./tokenbench-demo/suite.yaml --dry-run
uv run tokenbench baseline --suite ./tokenbench-demo/suite.yaml --out ./runs/baseline
```

The demo should report 36 passing trials and 36 attempts. Open `runs/baseline/summary.html` in a browser, or read `runs/baseline/summary.json`.

If your checkout is on exFAT, run `export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/tokenbench/venv"` before `uv sync`. Keep that environment setting when running the CLI so `uv` can use a filesystem that supports symlinks.

## CLI reference

| Command | Behavior |
| --- | --- |
| `tokenbench --help` | Lists the available commands. `tokenbench init --help`, `tokenbench doctor --help`, and `tokenbench baseline --help` show command options. |
| `tokenbench doctor` | Reports the Python and SQLite versions and performs an in-memory SQLite write. |
| `tokenbench init --path DIRECTORY` | Copies `suite.yaml`, `cases.jsonl`, and `prices.json` to the requested directory. Existing demo files cause the command to stop before copying. |
| `tokenbench baseline --suite FILE --dry-run` | Validates YAML, all task records, the price snapshot if configured, and the Python entrypoint. It imports the configured module. It runs zero trials and writes zero output files. |
| `tokenbench baseline --suite FILE --out DIRECTORY` | Runs the selected cases sequentially and writes the three artifacts below. `--out` must point to a new or empty directory. A completed benchmark exits successfully regardless of the measured pass rate. Invalid configuration and filesystem errors produce a nonzero exit status. |

Run the commands with `uv run` from the project root, or directly in the installed environment.

## Connect a Python application

Install the target application in the same Python environment as TokenBench. Set `entrypoint` to an importable `package.module:function`. The callable may be synchronous or asynchronous. It receives the case's `input` object and a `TraceSink` and returns a JSON-compatible answer. The grader compares object keys and values recursively, ignores object key ordering, and preserves array ordering and JSON types.

For example, wrap an existing lookup workflow with an importable `my_app.benchmark` module.

```python
from pydantic import JsonValue

from my_app.workflow import lookup
from tokenbench.trace import TraceSink


def run(task_input: dict[str, JsonValue], _trace: TraceSink) -> JsonValue:
    query = task_input["query"]
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    return lookup(query)
```

For usage accounting, pass the trace sink into the workflow and wrap each underlying model call. This gives failed calls and internal retries their own traces. Use the sink on the callable's thread and finish instrumented work before returning. `call.report_usage(ProviderUsage(...))` accepts normalized provider data. Import `ProviderUsage` from `tokenbench.pricing`. Pass `input_tokens`, `output_tokens`, and, when available, `cached_input_tokens`, `reasoning_tokens`, or `cost_usd`. Counts and cost can be omitted when unknown. Use `call.estimate_usage(input_tokens, output_tokens)` for local tokenizer or heuristic counts. `trace.model_call(provider=..., model=...)` can override the suite's identifiers for that call.

```python
with trace.model_call() as call:
    result = model.generate(prompt)
    call.report_usage(
        ProviderUsage(
            input_tokens=result.prompt_tokens,
            output_tokens=result.completion_tokens,
        )
    )
```

An uninstrumented callable can still run and be graded. Its attempts have no model spans and the corresponding token and cost totals are incomplete. Use trusted targets and configuration. Imports and execution use the current process's permissions. The runner passes case inputs to the target and keeps expected answers in the grader. Ledger and report data include identifiers, counts, timing and exception types. Prompt content, completions, expected answers, and exception messages are excluded.

Save the suite below as `suite.yaml`. File locations are relative to that YAML file. Omit `split` to include all cases, or use `dev`, `validation`, or `test`. Repetitions default to 1 and retries to 0. A retry follows a target or output validation exception. A completed but incorrect answer is graded once per repetition.

```yaml
schema_version: 1
suite_id: inventory-lookup
target:
  adapter: python
  entrypoint: my_app.benchmark:run
  provider: local
  model: inventory-v1
cases: cases.jsonl
split: dev
repetitions: 1
max_retries: 1
```

Each line in `cases.jsonl` holds one task with a unique `task_id`. This example expects the workflow to return the object shown in `expected`.

```json
{"schema_version":1,"task_id":"oak-code","input":{"query":"What is Oak Point's catalog code?"},"task_type":"lookup","split":"dev","grader_spec":{"kind":"exact","expected":{"answer":"OP-201"}}}
```

Optionally add `price_snapshot: prices.json` to the suite to estimate USD cost from frozen per-million input and output rates. The bundled `prices.json` shows the v1 format. Provider-reported cost takes precedence. Provider-reported token counts take precedence over estimates. Reasoning and cache-read counts, when supplied, are treated as subsets of output and input respectively. A call with known cached input and no reported USD cost has unknown cost because the snapshot only supplies standard rates.

## Run artifacts

- `baseline.sqlite3` stores the manifest, every started attempt, and every instrumented model call. Each attempt records its repetition, retry number, status, timestamps, and elapsed time. A process interruption can leave `running` rows for work that started and could not be finalized.
- `summary.json` contains the manifest, paired task and repetition identifiers, final-trial success totals, all attempts, and per-call usage and cost sources. Totals include failed calls and retries. A complete total is `null` when relevant observations are missing. Known subtotals and missing counts are shown separately. USD amounts are decimal strings.
- `summary.html` presents the run metadata, aggregate results, and attempt ledger as a standalone report.

The demo's `local_fixture` label identifies an end-to-end fixture evaluation. Its whitespace-based counts are estimates, and its zero USD rates apply only to the local fixture model. Repeating a frozen run to a fresh directory produces identical JSON and HTML. Observed timing stays in SQLite. The manifest fingerprints the suite, dataset, price snapshot when present, and the callable's source module when available.

## Development checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```
