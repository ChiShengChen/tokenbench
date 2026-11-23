# TokenBench

TokenBench runs local baselines and deterministic policy experiments for Python applications. It executes each case against a callable, grades its JSON result by exact match, and records attempts, model calls, usage, and policy events in SQLite. The bundled toy RAG suite runs end to end against frozen records and a deterministic fixture model.

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
| `tokenbench --help` | Lists the available commands. Add `--help` after a command to show its options. |
| `tokenbench doctor` | Reports the Python and SQLite versions and performs an in-memory SQLite write. |
| `tokenbench init --path DIRECTORY` | Copies `suite.yaml`, `cases.jsonl`, `prices.json`, and `experiment.yaml` to the requested directory. Existing demo files cause the command to stop before copying. |
| `tokenbench baseline --suite FILE --dry-run` | Validates YAML, all task records, the price snapshot if configured, and the Python entrypoint. It imports the configured module. It runs zero trials and writes zero output files. |
| `tokenbench baseline --suite FILE --out DIRECTORY` | Runs the selected cases sequentially and writes the three artifacts below. `--out` must point to a new or empty directory. A completed benchmark exits successfully regardless of the measured pass rate. Invalid configuration and filesystem errors produce a nonzero exit status. |
| `tokenbench run --experiment FILE --dry-run` | Validates the experiment, suite, Python entrypoint, and enabled policy parameters, order, and hook requirements. It runs zero trials and writes zero output files. |
| `tokenbench run --experiment FILE [--out DIRECTORY]` | Runs the baseline and named candidates on the same cases. Output defaults to `runs/<experiment-id>` and must point to a new or empty directory. Candidate pass rates and coverage are reported as results. |

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

## Run a policy experiment

```sh
uv run tokenbench run --experiment ./tokenbench-demo/experiment.yaml --dry-run
uv run tokenbench run --experiment ./tokenbench-demo/experiment.yaml --out ./runs/topk
```

Use `init` with a fresh directory if your demo predates `experiment.yaml`. The bundled experiment selects the first two or first one of the three ranked documents. The frozen fixture passes 36/36 cases in all three configurations and emits 1,776 estimated total tokens for baseline, 1,344 for top-2, and 912 for top-1.

An experiment YAML contains `schema_version: 1`, an `experiment_id`, a `suite` path relative to the YAML, and a `candidates` list. Each candidate has a unique `name` and an ordered `policies` list. Each policy has an `id`, optional `hook` and `enabled` fields, and typed `parameters`. Set `enabled: false` to disable a policy or use `policies: []` for a no-op candidate. This example uses the bundled suite.

```yaml
schema_version: 1
experiment_id: toy-rag-single
suite: suite.yaml
candidates:
  - name: topk-2
    policies:
      - id: rag.topk
        parameters: {k: 2}
```

See all seven implemented policy IDs, parameters, regression examples, and researched future methods in the [strategy catalog](docs/strategy-catalog.md).

### Add target hooks

Declare the hooks your target invokes under `target.hooks` in the suite YAML. A Python target calls `trace.apply_context` with typed segments and uses the returned sequence to assemble its prompt or next input. For example, after ranking a set of retrieved documents, pass each document as a segment.

```python
from tokenbench.policies.models import ContextSegment

documents = trace.apply_context(
    "retrieval",
    [
        ContextSegment(
            segment_id=doc.source_id,
            kind="retrieved_doc",
            content=doc.text,
            source_ids=(doc.source_id,),
            mutable=True,
        )
        for doc in ranked_documents
    ],
)
prompt_context = [segment.content for segment in documents]
```

Use the `tool_response` hook for tool payloads and `history` for chronological history blocks. History blocks with the same `turn_id` form a complete turn. Use `trace.step()` immediately before each agent step. Segment contents are immutable by default. Mark only optional data `mutable=True`, pin cited evidence with `protected=True`, and use `exact=True` for byte-sensitive content. Instructions, the active query, code, schemas, commands, credentials, citations, and system or developer roles receive automatic protection. For projection, set `required_keys` on a structured segment to preserve additional exact fields. Use opaque IDs in `segment_id`, `source_ids`, and artifact references because the ledger stores these identifiers.

For log trimming, pass a `tool_result` segment with `format="log"`. Its JSON content must include `lines`, `diagnostic_lines` with zero-based indexes into the original lines, `exit_code`, and `artifact_ref` for the complete frozen log. Mark every known failure cause and stack-trace line as diagnostic. The runner retains explicitly marked diagnostics, recognizable error markers, and the remainder of the log after a recognizable stack start. An unclassified failed log causes a policy failure. The complete log stays at the target-provided artifact reference.

Experiments use the same suite and input order for all candidates. Reset any external target state between trials as part of the adapter. A candidate is marked `unsupported` when it declares an enabled hook that is never observed across its cases. A policy contract violation is audited as `failed`. A step cap records a failed trial with a `blocked` policy event.

## Run artifacts

- `baseline.sqlite3` stores the manifest, every started attempt, and every instrumented model call. Each attempt records its repetition, retry number, status, timestamps, and elapsed time. A process interruption can leave `running` rows for work that started and could not be finalized.
- `summary.json` contains the manifest, paired task and repetition identifiers, final-trial success totals, all attempts, and per-call usage and cost sources. Totals include failed calls and retries. A complete total is `null` when relevant observations are missing. Known subtotals and missing counts are shown separately. USD amounts are decimal strings.
- `summary.html` presents the run metadata, aggregate results, and attempt ledger as a standalone report.

An experiment creates `baseline/` and one directory per candidate with its own ledger and summaries. The experiment root contains `comparison.json` and `comparison.html` with paired regressions, improvements, and token and USD deltas. Unknown or unsupported deltas are `null` in JSON. The per-attempt policy audit includes changed or removed segment IDs, kept and removed references, estimated before/after context tokens, compressor overhead, and failure reasons. SQLite stores policy runtime, and the deterministic JSON includes policy IDs, version, order, and configuration hashes. Context estimates use whitespace counts for hook inputs and outputs. Step checks record their step number and leave context counts unset. End-to-end token deltas come from model call usage in the ledger.

The demo's `local_fixture` label identifies an end-to-end fixture evaluation. Its whitespace-based counts are estimates, and its zero USD rates apply only to the local fixture model. Repeating a frozen run to a fresh directory produces identical JSON and HTML. Observed timing stays in SQLite. The manifest fingerprints the suite, dataset, price snapshot when present, and the callable's source module when available.

## Development checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```
