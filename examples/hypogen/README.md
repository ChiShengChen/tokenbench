# Hypogen evidence benchmark

This benchmark runs six fixed cases through Hypogen's research store and `RankedLive` adapter. `TraceSink` measures each serialized chat completion request and its mock response. HTTPX `MockTransport` handles requests in process, and Hypogen's `LinkedFixture` builds the responses from the received request. Each case has a temporary SQLite store. Evidence search is disabled, and the run requires no API key, network connection, or API spend.

## Run the comparison

Use a Hypogen checkout that contains `research_app` at the repository root and includes the ranked live adapter and its fixture files. The example was verified with Hypogen revision `20b1ed4`. Install TokenBench with `uv sync` and run the following from the TokenBench root. Set `HYPOGEN_ROOT` to your checkout path.

```bash
HYPOGEN_ROOT=/path/to/hypogen
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$HYPOGEN_ROOT" \
  uv run --offline --no-sync tokenbench run \
  --experiment examples/hypogen/experiment.yaml --out runs/hypogen-metered
```

The command runs the baseline and both candidates over the same six cases. Open `runs/hypogen-metered/comparison.html` for the paired comparison or `runs/hypogen-metered/pareto.html` for the trade-offs. Each arm has a SQLite ledger and a JSON and HTML summary. Use a new output path for another run.

| Arm | Evidence sent to the model |
| --- | --- |
| Baseline | All evidence fields prepared by Hypogen |
| `evidence-dedup` | One copy of each exact duplicate row. Identical text with distinct citations is retained. |
| `evidence-projection` | Drops `document_id` and `paragraph` when the citation ID encodes both unambiguously. |

The wrapper checks citation IDs, evidence text and provenance, publication and retrieval times, and all other model payload data on every call. It checks the point-in-time cutoff on the model request and stored report. The financial cases also check the model inputs' currency, unit and evidence citation before and after scenario recalculation. Grading covers EPS, break-even lots, scenario recalculation, citation availability at two times, missing supply evidence and the ASML historical cutoff.

The meter estimates tokens by counting ASCII alphanumeric runs and individual non-whitespace characters in the serialized request body and returned completion text. The request includes the system instructions, schema and user payload. Use the counts to compare these mock runs. Provider tokenization, cached reads, and production cost and latency require separate measurements.

## Run the integration test

```bash
TOKENBENCH_HYPOGEN_ROOT="$HYPOGEN_ROOT" PYTHONDONTWRITEBYTECODE=1 \
  uv run --offline --no-sync pytest tests/test_hypogen_example.py
```

The tests cover the meter and preservation checks locally. Setting `TOKENBENCH_HYPOGEN_ROOT` also enables the end-to-end comparison against that checkout.
