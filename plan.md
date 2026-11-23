# TokenBench: Reproducible Token-Efficiency Optimization for LLM and Agent Systems

> **Implementation-ready plan for Codex / engineering agents**  
> Status: Proposed · Version: 0.1 · Date: 2026-11-23  
> Working repository name: `tokenbench` (rename if unavailable)  
> Primary deliverable: an open-source, local-first benchmarking and optimization toolkit, **not** a hosted SaaS or a new prompt-compression model.

## 0. Mission and non-negotiable requirements

Build a Python toolkit that connects to an **existing target LLM application or multi-agent workflow**, captures its end-to-end traces, applies configurable combinations of *mature* token/cost optimization strategies, and discovers the best **quality-preserving** configurations under user-specified budgets.

The tool must answer:

1. What parts of the target workflow consume tokens and money (system/user/history, retrieved documents, tools, agent communication, completion/reasoning, retries, caches)?
2. Which individual techniques reduce **total end-to-end tokens**, **billed cost**, and/or **wall-clock latency** without unacceptable quality degradation?
3. Which **ordered combinations** of techniques work best, and when do techniques conflict or make one another redundant?
4. What is the **Pareto frontier** among task success, tokens, dollars, latency, and reliability? How robust is the result across task categories and repeated runs?
5. Can an independent engineer reproduce and audit the winning configuration, including **what was dropped or changed**?

### Non-goals (MVP)

- Training a novel compressor, fine-tuning a foundation model, or making unverifiable claims about compression rates.
- Building a general-purpose agent framework. Connect existing frameworks rather than replacing them.
- Treating reduced token count as equivalent to reduced bill or improved success rate.
- Optimizing against proprietary benchmark test answers or allowing benchmark answers to enter the agent context.
- Secretly modifying system/developer instructions, user queries, tool schemas, executable code, citations, credentials, or safety-relevant content.
- Claiming universal superiority from one model, dataset, or random seed.

## 1. Core design principles

**Correctness first.** Define a task success criterion before optimization. Compare against a frozen, unoptimized baseline. Never accept a cheap configuration that violates the quality floor.

**Measure full executions.** Include all LLM calls (including compressors, routers, judges, rewriters), retries, tools, overhead, cache writes and reads, and failures. Record token savings and dollar savings separately.

**Compare like with like.** Use paired tasks, model/agent versions, tool fixtures, dataset partitions, seeds when supported, concurrency limits, price snapshots, and explicit cache conditions.

**Keep protected information intact.** Compression may operate on selected low-trust, content-heavy segments but not protected instructions, exact tool arguments or identifiers. Preserve citations and provenance.

**Make comparisons composable, but not naïve.** Policies have stages and prerequisites. The search engine cannot arbitrarily permute incompatible transforms, and it must retain policy order in the run ID.

**Local-first and reproducible.** SQLite/DuckDB + Parquet + CLI first; optional web dashboard, Postgres, managed telemetry, distributed workers and cloud benchmarks later.

## 2. User experience / expected commands (CLI contract)

Commands below are target interfaces; implement them incrementally rather than assuming they already exist.

```bash
# 1. Initialize project and demo benchmark
tokenbench init --path ./tokenbench-demo
tokenbench doctor

# 2. Register an existing agent (Python callable / HTTP API / OpenAI-compatible)
tokenbench target add --config configs/targets/my_agent.yaml

# 3. Inspect baseline cost and traces
tokenbench baseline --suite configs/suites/my_workflow.yaml --out runs/baseline

# 4. Test named single strategies and combinations
tokenbench run --experiment configs/experiments/screening.yaml

# 5. Search ordered combinations under a spend cap
tokenbench optimize --experiment configs/experiments/optimize.yaml --budget-usd 25

# 6. Make paired comparisons and generate an auditable report
tokenbench compare --baseline runs/baseline --candidates runs/optimize --out reports/summary.html
tokenbench export --run <run-id> --format jsonl

# 7. CI: block a change if quality worsens or cost per success rises
tokenbench gate --experiment configs/experiments/ci.yaml
```

A user should be able to start with **one Python target adapter, ten task cases, a deterministic transformation, and an objective grader** without cloud infrastructure.

## 3. System architecture

```text
External target system (LangGraph / custom Python / API / coding agent)
       |                         
       v
TargetAdapter + TraceCollector ---> Redaction / Provenance / Frozen replay fixtures
       |                                    |
       v                                    v
ExperimentRunner <--- Dataset + Grader + Baseline + Pricing snapshot
       |
       v
Policy compiler (stages, type checks, protected segments, constraints)
       |
       +--> deterministic filters / schema projectors / dedup / retrieval
       +--> optional LLMLingua / history memory compaction
       +--> cache-aware prompt assembly / model routing / step budgets
       |
       v
Execution + Verification + Usage accounting (all calls, failures, caches)
       |
       v
Result store (SQLite + Parquet/artifacts) -> Statistics / Pareto / Reports
       ^
       |
Search engine (screening -> adaptive search -> confirmation on holdout)
```

**Two execution modes must be kept distinct:**

1. **Offline trace-transform diagnostic:** recompute token counts, contextual relevance, and loss indicators from a captured trace, or replay an explicitly mocked/frozen completion. **No claim of end-to-end task success** from offline token counts alone.
2. **Online end-to-end evaluation:** run the modified target on task inputs through the same sandbox/environment and trusted grader. This is the only basis for quality-constrained optimization claims.

## 4. Repository layout

```text
tokenbench/
├── README.md
├── plan.md
├── pyproject.toml
├── LICENSE
├── .github/
│   ├── workflows/ci.yml
│   └── ISSUE_TEMPLATE/
├── src/tokenbench/
│   ├── cli.py
│   ├── config.py                 # versioned Pydantic models
│   ├── targets/
│   │   ├── base.py               # target interface and capabilities
│   │   ├── python_callable.py
│   │   ├── http.py
│   │   ├── openai_compatible.py
│   │   └── langgraph.py          # optional integration
│   ├── trace/
│   │   ├── schema.py
│   │   ├── capture.py
│   │   ├── usage.py
│   │   ├── sanitize.py
│   │   └── replay.py
│   ├── policies/
│   │   ├── base.py
│   │   ├── compiler.py
│   │   ├── deterministic.py
│   │   ├── retrieval.py
│   │   ├── llmlingua.py          # lazy extra
│   │   ├── memory.py
│   │   ├── caching.py
│   │   ├── routing.py
│   │   └── control.py
│   ├── datasets/
│   │   ├── schema.py
│   │   ├── jsonl.py
│   │   └── integrations/          # optional benchmark-specific adapters
│   ├── graders/
│   │   ├── exact.py
│   │   ├── json_schema.py
│   │   ├── tests.py
│   │   └── llm_judge.py           # optional; audited, counted separately
│   ├── experiments/
│   │   ├── runner.py
│   │   ├── manifest.py
│   │   ├── budgets.py
│   │   ├── randomization.py
│   │   └── scheduler.py
│   ├── optimizers/
│   │   ├── screening.py
│   │   ├── grid.py
│   │   ├── optuna_search.py
│   │   └── pareto.py
│   ├── metrics/
│   │   ├── aggregate.py
│   │   ├── statistics.py
│   │   └── pricing.py
│   ├── storage/
│   │   ├── sqlite.py
│   │   └── artifacts.py
│   └── reports/
│       ├── html.py
│       ├── csv.py
│       └── plots.py
├── configs/
│   ├── targets/
│   ├── suites/
│   └── experiments/
├── examples/
│   ├── toy_rag/
│   ├── toy_multi_agent/
│   └── custom_target/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── regression/
├── docs/
│   ├── target-adapter.md
│   ├── strategy-catalog.md
│   ├── benchmark-protocol.md
│   ├── security.md
│   └── reproducibility.md
└── reports/                       # gitignored generated data
```

Python >=3.11, `uv` or pip, Pydantic v2, Typer, pytest, pandas/polars, DuckDB or SQLite, Plotly or matplotlib, Jinja2, Optuna. Optional extras for `llmlingua`, `langgraph`, `litellm`, `inspect-ai`, `datasets`, and rerankers. Pin and lock versions. Avoid requiring all heavy extras for basic operation.

## 5. Typed contracts and trace instrumentation

### 5.1 Core records

Implement versioned schemas (Pydantic or dataclasses):

- `TaskCase`: `task_id`, `input`, `task_type`, `grader_spec`, `split` (`dev|validation|test`), `tags`, `fixture_refs`, optional `expected_artifacts` and privacy metadata.
- `TargetSpec`: adapter kind, entrypoint, model/provider identifiers, declared capabilities (`supports_messages`, `supports_tool_hooks`, `supports_route`, `supports_cache`, `supports_model_usage`, `supports_seed`, `supports_sandbox`), tool environment.
- `ContextSegment`: `segment_id`, `role`, `type` (`instruction|query|history|retrieved_doc|tool_result|agent_message|code|schema|other`), content, source IDs, `mutable`, `protected`, token count, transformations, exactness requirement.
- `PolicySpec`: stable ID/version, stage, required capabilities, parameters, constraints, supported segment types, cost model.
- `ExecutionTrace`: trace/parent/span IDs, task, model, agent, tool names, prompt segments, redacted inputs/outputs, start/end, status, retries, hashes, errors.
- `UsageRecord`: provider/model, prompt tokens, completion tokens, reported reasoning tokens (if available), cached read/write tokens, provider-reported billed usage, tool units, estimated/actual USD, currency, `usage_source`, timestamps.
- `EvaluationRecord`: objective score, pass/fail, optional judge score, grader version, evidence/provenance, error categories.
- `RunManifest`: content hashes of dataset and policies, source commit, config, dependency lockfile hash, pricing snapshot, model IDs/versions, timestamp, seeds, cache regime, environment, sampler, budget.

Every policy must be able to report `before_tokens`, `after_tokens`, `segments_modified`, `preserved_references`, `runtime_ms`, `compressor_api_tokens/cost`, and a failure reason. Do not store secrets in plaintext traces; support configurable truncation and hashed payloads.

### 5.2 Hooks and integration boundaries

Prefer **non-invasive OpenTelemetry/OpenInference-compatible tracing** when the target exposes spans. Provide Python hooks for `before_model_call`, `after_model_call`, `before_tool_response`, `after_tool_response`, `before_agent_handoff`, `after_agent_handoff`, and `before_history_update`.

Do not promise interventions for opaque third-party systems: if only an HTTP final response is exposed, record the outer request and provider usage if available and mark inside-agent policies **unsupported**. If no provider usage is available, tokenizer-based estimates must be labeled estimates. No intercepted message modification without the target's explicit integration/opt-in.

### 5.3 Token and dollar accounting

Track independently:

- Raw user-facing workload tokens (baseline-comparable context, pre-optimization)
- Actual prompt tokens received by model; output and reasoning tokens when reported
- Billable cached/non-cached tokens, cache writes, read discounts, and expiration policy
- Auxiliary LLM tokens for routing, compression, summarization, verification, judging
- API/model cost + tool/search/reranking/compute cost, with unknown costs separately flagged
- Retry/failed-call costs; tokens and cost **per full attempt** and **per successful task**

Provider usage records take precedence over tokenizer estimates. Avoid double-counting cached tokens already included in provider-reported `input_tokens`; normalize vendor-specific schemas. Track cost versions, billing tier, currency, and whether data is estimated. **A cache hit can reduce dollars without reducing prompt tokens.** If a provider exposes only limited reasoning usage, record `unknown` rather than zero.

## 6. Strategy catalog (mature/common methods first)

All policies use `Policy.apply(context, task, runtime_state) -> PolicyResult`. Multiple components are allowed within a stage, but explicitly declare commutativity, dependency, and forbidden pairings. The default must be **no-op**.

| ID | Stage | Strategy / implementation | Search parameters | MVP |
|---|---|---|---|---|
| `filter.boilerplate` | tool/post-fetch | Strip HTML boilerplate, ANSI escapes, redundant metadata; preserve cited content | allowlist, max chars | P0 |
| `filter.schema_projection` | tool | JSONPath/key allowlist on structured tool responses | field paths, depth | P0 |
| `filter.deduplicate` | tool/history | Exact-hash or normalized block dedup, provenance retained | granularity, normalize | P0 |
| `filter.log_tail` | tool | Preserve error lines + first/last N lines, overflow pointer | head/tail N, max lines | P0 |
| `history.window` | history | Keep latest N turns plus pinned state | turn count/token budget | P0 |
| `history.summary` | history | Summarize completed subtask/history, keeping requirements & decisions | model, trigger, token cap | P1 |
| `history.state_compaction` | history | Structured state snapshot + evidence references | schema, cadence | P1 |
| `rag.topk` | retrieval | Adjust retrieved chunk count | k = 2/4/8/16 | P0 |
| `rag.bm25` | retrieval | Lexical retrieval/filter | k | P1 |
| `rag.rerank` | retrieval | Cross-encoder or existing reranker | k, threshold | P1 |
| `rag.mmr` | retrieval | Diversity-aware reranking | lambda, k | P1 |
| `compress.llmlingua2` | context | Learned token retention on permitted context | rate = .25/.5/.75 | P1 |
| `compress.longllmlingua` | context | Query-aware compression | rate, reorder options | P2 |
| `prompt.static_prefix` | assembly | Stable prefix, reusable template layout | prefix layout | P1 |
| `cache.provider` | runtime | Supported provider prompt caching, explicit cold/warm regimes | policy, TTL | P1 |
| `cache.response` | runtime | Exact/semantic response cache **only for safe tasks** | exact or semantic; TTL | P2 |
| `model.route` | orchestration | Rule-based cheap/strong model selection | routes, confidence | P1 |
| `agent.step_budget` | control | Limit steps with fail-safe/abstain, not silent truncation | max steps | P0 |
| `agent.duplicate_tool` | control | Avoid repeated identical pure/read-only tool calls | scope, TTL | P1 |
| `agent.batch` | control | Batch compatible independent calls where target permits | batch size | P2 |
| `agent.handoff_schema` | communication | Compact JSON state and evidence IDs for inter-agent messages | schema, fields | P2 |
| `prompt.dspy` | offline optimization | DSPy instructions/examples optimizer; frozen prompt per trial | optimizer, budget | P2 |

**Important technical guardrails**:

- `filter.log_tail` must preserve stack traces, failure causes, exit code and a reference to the complete artifact. Never rewrite code or command outputs in a way that fakes success.
- Protect system/developer instructions, current user query, tool contracts, validation schemas, executable commands, citation IDs, numerical constraints, secrets and user-marked spans from lossy compression.
- For `compress.llmlingua2`, run the real upstream library behind optional dependencies; no pretend implementation named LLMLingua. Log the model checkpoint and overhead.
- Retrieval/reranking require a corpus and relevance evaluation; they do not operate automatically on arbitrary agent history.
- `cache.response` can produce semantically wrong or stale results; default **disabled**, require explicit data-sensitivity and freshness checks, separate warm and cold runs.
- `model.route` changes the model, and thus introduces a confound; report model-constant experiments separately from routing experiments.
- Timeout, step-budget, and duplicate-call policies can increase failures; include these in intent-to-treat outcomes.
- Not all techniques are plug-and-play. Declare `unsupported`, `skip`, or `dependency_unavailable`, never count them as successful trials.

## 7. Composition / experiment search

### 7.1 Policy pipeline graph

Canonical default order:

```text
retrieve -> coarse filtering / top-k -> rerank -> optional query-aware compression
-> prompt/history assembly & optional compaction -> stable-prefix layout
-> model routing + provider caching -> LLM/tool execution
-> tool-output filtering / duplication control -> verification / repeat / finish
```

Some stages recur inside the agent loop. Represent the policy as a directed acyclic *stage* graph, plus loop-local hooks. The compiler validates ordering and mutability boundaries before scheduling a run. Example: `rerank` needs retrieved candidates; `cache.provider` relies on stable prompt bytes; `history.summary` must be applied on correct history intervals; cache changes can affect compaction benefits.

### 7.2 Search algorithm (avoid naïve full Cartesian product)

A 15-method search with several hyperparameters can explode into thousands of expensive runs. Implement:

1. **Baseline:** no transformations; >=3 repeat runs on stochastic tasks if affordable.
2. **Stage 1 – low-cost screening:** one-factor-at-a-time (OFAT), deterministic policies first; quick dev subset; check eligibility and protected spans.
3. **Stage 2 – interactions:** pairwise or covering-array tests across top methods (including deliberately incompatible/negative interactions), varying the **order** only for declared order-sensitive pairs.
4. **Stage 3 – adaptive multiobjective search:** Optuna TPE / NSGA-II for eligible policy choices and hyperparameters. Apply hard quality gates and cap per-trial/token/USD budgets. Optionally successive halving with caution about noisy rankings.
5. **Stage 4 – confirm:** replay winners and baseline on untouched holdout; repeated paired trials, cold and warm cache strata, stratify by task class.
6. **Stage 5 – export:** best feasible configuration, Pareto set, interaction chart, failed/unsupported results, uncertainty intervals, reproducible manifest.

Avoid using the test set for policy selection; separate `dev` (engineering), `validation` (optimization) and **sealed test** (one-shot confirmation) splits. Define a no-data-leakage policy. Randomize execution order within matched blocks to reduce provider drift or cache warming confounds. Respect API rate limits and shared quotas.

### 7.3 Objective and constraints

Main optimization:

```
minimize expected_total_cost_usd / expected_successes
subject to:
  paired_quality_drop <= epsilon (e.g. 0.02 absolute, user configurable)
  lower_confidence_bound_quality >= minimum_quality  # optional stricter gate
  p95_latency_ms <= budget_p95_ms                   # optional
  max_spend_usd <= experiment_budget_usd
  all protected-span and security checks pass
```

A practical implementation should treat a zero-success candidate as `cost_per_success = +inf`, use observed total cost / number of successes over the sample (not the average of per-task cost/success), and report uncertainty rather than pretending the ratio is precise. Add a tunable multiobjective view (`success_rate ↑`, `cost/task ↓`, `tokens/task ↓`, `p95_latency ↓`, `failures ↓`). Do **not** collapse everything into one arbitrary scalar: provide a Pareto frontier and allow users to select their quality floor and budget.

Report both per-attempt and per-success values; a configuration that saves 60% prompt tokens but doubles retries may have worse aggregate outcomes.

### 7.4 Confidence intervals / statistical tests

- Paired task-level differences: bootstrap by **task** (respect grouping/repeated trials), report 95% confidence intervals.
- Binary success: paired discordant outcomes, McNemar-style comparison where valid, plus repeated-run hierarchical bootstrap as needed.
- Avoid treating each agent step as an independent observation when the experimental unit is a task.
- Display dataset size, number of runs, model and grader versions, price snapshot, significance caveats, practical effect size.
- Mark exploratory results versus confirmatory test results. No fabricated p-values or savings percentages.

## 8. YAML configuration format (v1 proposal)

```yaml
schema_version: 1
experiment_id: sample-token-efficiency
mode: online                 # online | offline_diagnostic
target:
  adapter: python
  entrypoint: examples.toy_rag.agent:run
  model: ${TB_MODEL}         # resolved from environment, never persist secret
suite:
  source: configs/suites/toy_rag.yaml
  split: validation
  max_cases: 40
baseline:
  policy: []
  repetitions: 3
accounting:
  cost_source: provider_then_snapshot
  include_auxiliary_cost: true
  price_snapshot: configs/pricing/prices.lock.json
  cache_regimes: [cold, warm]  # separate results, never pool silently
policies:
  candidates:
    - id: filter.schema_projection
      parameters: {allow_keys: [title, source_id, snippet, url]}
    - id: filter.deduplicate
      parameters: {granularity: block}
    - id: rag.topk
      parameters: {k: [2, 4, 8]}
    - id: rag.rerank
      parameters: {k: [2, 4], min_score: [0.0, 0.3]}
    - id: history.window
      parameters: {max_turns: [4, 8]}
    - id: compress.llmlingua2
      enabled: false          # opt-in, heavyweight dependency
      parameters: {rate: [0.25, 0.50, 0.75]}
    - id: agent.step_budget
      parameters: {max_steps: [8, 16]}
search:
  algorithm: screening_then_optuna
  max_trials: 40
  seed: 42
  paired_task_order: true
  max_concurrency: 2
  max_cost_usd: 25.0
  early_stop_on_failed_protected_spans: true
quality:
  metric: task_success
  max_absolute_drop: 0.02
  min_success_rate: 0.85
  grader: deterministic_then_optional_judge
objectives:
  - total_cost_per_success
  - total_tokens_per_success
  - p95_latency_ms
report:
  outputs: [html, json, csv, parquet]
  show_pareto: true
  show_per_task_deltas: true
  show_interactions: true
```

All YAML is validated before execution; invalid policy parameter types, unresolvable entrypoints and unsupported hooks must fail clearly. The example above is a **schema target**, not proof every policy is implemented in milestone 1.

## 9. Target adapters and controlled grading

### 9.1 Target contract

```python
class TargetAdapter(Protocol):
    def capabilities(self) -> set[str]: ...
    async def run(
        self, task: TaskCase, policy: CompiledPolicy, trace: TraceSink
    ) -> TargetResult: ...
```

Implement adapters in this order:

1. **Python callable** with optional model/tool hooks; suitable for toy apps and a custom agent.
2. **HTTP API** for black-box baseline/output/latency metrics; do **not** claim internal intervention access.
3. **OpenAI-compatible** model wrapper / LiteLLM integration for request mutation and usage logging.
4. **LangGraph** messages/history/node hook integration via optional extras.
5. Later: **MCP tool output** hooks, coding-agent wrappers, OpenTelemetry/OpenInference trace ingestion.

### 9.2 Graders

Prioritize deterministic, environment-based grading: exact answer match, structural JSON schema, assertion checks, executable unit tests, tool state, reference evidence recall. LLM-as-judge is optional and should use a fixed rubric, blind policy labels, record grader model/seed/version and count its costs separately under **evaluation overhead**, not production run cost. Human review is required for safety-critical/ambiguous regressions.

### 9.3 Fixture safety

Freeze tool outputs for reproducible integration tests; distinguish fixture replay from real live tools. Sandbox any code execution, disable network unless necessary, and explicitly approve any real-world write action. Default benchmarks must not send emails, place orders, submit trades, or mutate production systems. Clean environments between trials to prevent state leakage.

## 10. Benchmark suites

### Ship locally with the project (must work offline)

- **Toy-RAG**: 30–100 curated cases with short/long docs, answer provenance, distractors, citations; vary context length and answer locations.
- **Toy-tool-use**: structured JSON responses with irrelevant columns, repeated tool calls, noisy logs; unit-test verified task state.
- **Toy-memory**: multi-turn tasks requiring remembering user constraints, revisions, temporal facts, and exact identifiers.
- **Toy-multi-agent**: researcher → analyst → verifier flow; structured handoffs; evidence checks and branch coverage.
- **Adversarial retention checks**: long-tail details, numbers, negation, identifiers, contradictory updates, citation integrity, required safety constraints.

### Optional external benchmark integrations (separate adapters; never copy copyrighted/proprietary test data into the repository)

| Benchmark | Use | Integration note |
|---|---|---|
| [LongBench](https://github.com/THUDM/LongBench) | Long-context QA and compression | Honor dataset license and exact scorer |
| [RULER](https://github.com/NVIDIA/RULER) | Long-context retrieval stress tests | Fixed lengths/positions |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | Multi-session memory | Test selective forgetting and revisions |
| [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) | Agentic long-term memory | Measure memory query latency |
| [SWE-bench Verified](https://www.swebench.com/verified.html) | Coding agents | Separate Docker/sandbox runner; use official scorer |
| [Terminal-Bench](https://www.tbench.ai/) | Terminal/tool-use agents | Match version and environment |
| [tau-bench](https://github.com/sierra-research/tau-bench) | Multi-step tool-use success | Preserve state/environment setup |

Do not ship heavy benchmark integrations as required installation dependencies. Respect each benchmark's licensing, contamination rules and scoring protocol. **SWE-bench leaderboard scores across different scaffolds are not a controlled compressor comparison**; our experiment must hold the agent scaffold constant.

## 11. Dashboards and required report views

The first deliverable is a standalone HTML report, plus JSON/CSV/Parquet data. Include:

1. **Run overview:** target/model/suite/grader, Git commit, policies/order, seed, cases, repetitions, cache regimes, total spend.
2. **Funnel / token attribution:** system/history/RAG/tool/handoff/input/output/auxiliary, cache read/write; per-agent and per-step when available.
3. **Single-method table:** baseline vs candidate on task success, input/output/total tokens, bill, cost per success, P50/P95 latency, failed runs; CIs.
4. **Pareto scatter:** cost vs quality, tokens vs quality, latency vs quality, hover to inspect policy composition.
5. **Interaction matrix:** incremental gain of pairs and ordered pairs vs expected independent gain; clearly identify observed samples.
6. **Per-case replay:** side-by-side original vs transformed segments (redacted), source provenance, score changes, tool traces, regression categories.
7. **Recommendation card:** best feasible candidate under specified constraints; show unsupported policies and uncertainty.
8. **Export policy:** exact YAML for winner plus lockfile/manifest and a CI regression threshold.

Use browser-local HTML with no public upload required. Include `--redact` default and optional trace payload opt-in.

## 12. Repository security, privacy and anti-gaming

- No real API keys, tokens, credentials or user PII in code, benchmark fixtures, published results, or committed logs. `.env.example`; secrets injected at runtime.
- Default trace retention: metadata and redacted payloads. Explicit `--capture-content` with warning, encryption/storage retention options for sensitive contexts.
- Never allow the optimizer to alter the grader, benchmark answers, target tests, tool permissions or the scoring rubric.
- Keep optimization inputs separated from sealed test answer keys and test labels.
- Disable provider prompt logging where appropriate; document privacy and data residency considerations.
- Guard against prompt injection inside retrieved/tool content; content is data, not instructions to the benchmark runner.
- Cap spend, tokens, wall time, concurrency and tool calls. On budget exhaustion, produce partial results marked incomplete, never a spurious winner.
- Pin datasets, dependencies, model versions when possible; detect changed remote model aliases and record uncertainty.

## 13. MVP plan and checkpoints (execute in sequence)

### Milestone 0 — Foundation / fixtures

**Deliver:** package + Typer CLI + typed config + SQLite store + pytest/ruff/mypy + `toy_rag` example + `tokenbench doctor`.

**Acceptance criteria:** `uv sync`, unit tests, CLI help, dry-run config validation and a no-API offline demo work on a clean machine. Document supported Python versions.

### Milestone 1 — Trustworthy baseline

**Deliver:** Python-callable adapter; task/grader schema; trace collection and pricing normalization; 30+ toy cases; baseline runner; deterministic JSON and HTML summary.

**Acceptance criteria:** at least one real end-to-end target runs; all attempted trials (including exceptions/retries) appear in the ledger; provider usage vs estimates are distinguishable; re-run with frozen fixtures gives deterministic structural results.

### Milestone 2 — First real optimizers

**Deliver:** schema projection, boilerplate filtering, deduplication, log-tail, top-k, history window, step budget; policy compiler and protected-span tests.

**Acceptance criteria:** every policy has unit tests, at least one negative test (quality break), before/after token counts, and audit log. Every policy can be disabled without changing baseline behavior.

### Milestone 3 — Combinations, search and statistical reports

**Deliver:** OFAT screening, pairwise/ordered-pair comparisons, basic Optuna search, spend/time budgets, bootstrap confidence intervals, Pareto chart, winner YAML.

**Acceptance criteria:** experiment with 3+ policy families executes within configured budget; clearly shows one case where greater compression worsens task success; winner survives validation gate; test split held out.

### Milestone 4 — Production integrations

**Deliver:** LiteLLM/OpenAI-compatible and LangGraph adapters, cache-aware accounting, LLMLingua-2 and reranker optional extras, provider cold/warm cache experiments, OpenTelemetry compatible traces where possible.

**Acceptance criteria:** end-to-end run on an example LangGraph agent; non-invasive black-box API adapter marks unsupported internal policies; token and cost metrics do not double count cached tokens; optional ML strategies report their overhead.

### Milestone 5 — External benchmarks and release quality

**Deliver:** at least one LongBench/LongMemEval adapter, one tool-use benchmark adapter, GitHub Actions CI regression gate, documentation, privacy policy, reproducibility guide, public sample results.

**Acceptance criteria:** a new contributor can reproduce a benchmark on approved fixture/sample split without keys, and run paid evaluations with their own provider credentials. Publish comparative results only with manifests, sample sizes and caveats.

### Future / research-track extensions (out of MVP)

- Cost-aware learned routing, bandit selection of context budgets, DSPy optimization.
- Agent-to-agent structured communication and minimum sufficient evidence/state.
- Uncertainty-aware retention, counterfactual missing-evidence analysis and error propagation.
- Distributed experiment scheduler; online shadow evaluation; human/LLM adjudication workflows.
- Direct benchmarking against alternative inference engines; disaggregate KV memory, throughput and compute cost from API token cost.

## 14. Agent execution instructions (for Codex)

Follow the milestones in order. Each PR must include:

1. Files changed and rationale, CLI examples, test instructions and test output.
2. Reproducible fixtures and fixed random seeds where applicable.
3. Explicit list of implemented vs planned policies: **no fake adapters or silent no-op optimization**.
4. Correctness, security, and unexpected cost/behavior considerations.
5. Documentation for how to connect one real existing agent system.

**Begin implementation without asking for clarification:** choose Python 3.11+, `uv`, Typer, Pydantic, SQLite, pytest, and a plain HTML report as defaults. Use a fake deterministic model for the initial offline smoke test. Do not make live paid API calls without an explicitly configured credential and an explicit CLI cost cap. Do not create or publish a GitHub repository or push commits until authorized in the actual working environment.

**First implementation task:** complete Milestone 0, then create the smallest Milestone 1 vertical slice: `TaskCase -> PythonCallableTarget -> TraceSink -> exact grader -> SQLite -> HTML/JSON report`. Add two deterministic policies to compare against the baseline. Do not attempt the entire roadmap in one PR.

## 15. Definition of done / release checklist

- [ ] The toolkit accepts arbitrary custom target through a documented adapter without modifying the target's evaluation criteria.
- [ ] Runs baseline and >=3 meaningful strategy combinations on identical task cases.
- [ ] Reports **total** token, bill, quality, retries, latency, and auxiliary optimization cost (with missing-data flags).
- [ ] Discriminates cached token **price reduction** from actual token **count reduction**.
- [ ] Rejects forbidden compression of protected segments and flags unsupported interventions.
- [ ] Produces a reproducible experiment manifest and machine-readable result dataset.
- [ ] Runs spend-capped search and returns feasible Pareto candidates with uncertainty.
- [ ] Identifies an example where fewer tokens lead to worse overall outcome.
- [ ] CI runs with no external API keys using deterministic fixtures.
- [ ] Documentation clearly distinguishes estimated, offline-diagnostic and online-measured results.

## 16. Useful upstream references (verify versions/API compatibility before implementation)

- Microsoft LLMLingua: https://github.com/microsoft/LLMLingua
- LiteLLM: https://docs.litellm.ai/
- LangGraph: https://docs.langchain.com/oss/python/langgraph/overview
- DSPy optimizers: https://dspy.ai/learn/optimization/optimizers/
- UK AISI Inspect AI: https://inspect.aisi.org.uk/
- OpenTelemetry: https://opentelemetry.io/
- Langfuse: https://langfuse.com/docs
- promptfoo: https://www.promptfoo.dev/docs/
- Optuna multi-objective optimization: https://optuna.readthedocs.io/
- LongBench: https://github.com/THUDM/LongBench
- RULER: https://github.com/NVIDIA/RULER
- LongMemEval / V2: https://github.com/xiaowu0162/LongMemEval , https://github.com/xiaowu0162/LongMemEval-V2
- SWE-bench Verified: https://www.swebench.com/verified.html
- Terminal-Bench: https://www.tbench.ai/
- tau-bench: https://github.com/sierra-research/tau-bench

**Research-quality reporting rule:** never borrow a compression percentage, leaderboard score, or cost saving from upstream as if it were measured on the user's target system. This project exists to measure the user's target under controlled conditions.
