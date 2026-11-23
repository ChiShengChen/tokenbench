# Token-efficiency strategy catalog

The catalog covers filtering, retrieval, context compression, memory, caching, routing, and prompt optimization. Each entry identifies what to measure on the same target and task cases. Use the objective grader to check for regressions before interpreting token or cost reductions.

## Implemented in Milestone 2

| Policy | Hook | Parameters and behavior | Regression to test |
| --- | --- | --- | --- |
| `filter.schema_projection` | `tool_response` | `allow_keys` keeps named top-level fields in mutable JSON tool results. Citation, error, command, schema, and other reserved fields are retained. The target can pin additional `required_keys`. | An omitted field contains the value the grader expects. |
| `filter.boilerplate` | `tool_response` or `retrieval` | `strip_ansi` defaults to `true` and removes ANSI SGR formatting. `drop_lines` lists exact whole lines to remove. This works for known navigation or footer lines in a feed with stable markup. | An approved line turns out to contain evidence, or a color distinction has semantic meaning. |
| `filter.deduplicate` | `tool_response` or `history` | `granularity: block` keeps the first structurally equal block and merges provenance. History deduplication stays within the same turn and role. | The number of repeated events is part of the answer. |
| `filter.log_tail` | `tool_response` | `head` and `tail` default to `2` and `10`. The policy keeps diagnostics, exit code, and the reference to the full artifact. Recognized stack starts retain the remaining lines. | An ordinary-looking progress line contains a value needed for grading. |
| `rag.topk` | `retrieval` | `k` keeps the first *k* ranked documents plus pinned evidence outside that prefix. | The only answer-bearing document ranks immediately below *k*. [Lost in the Middle](https://arxiv.org/abs/2307.03172) |
| `history.window` | `history` | `max_turns` keeps that many recent complete turns plus all pinned turns. Set it to `0` to keep just the pinned turns and other protected context. | An old preference or correction has no pin and falls outside the window. |
| `agent.step_budget` | `step` | `max_steps` caps calls to `trace.step()` before the agent starts each step. Exhaustion records an abstention as a failed trial. | The final verification takes one more step than the budget allows. [ReAct](https://arxiv.org/abs/2210.03629) |

The compiler checks stage order within each hook. Tool filtering and projection precede deduplication. Retrieval boilerplate filtering precedes top-*k*. History deduplication precedes the window. Independent tool filters can appear in either order. A disabled policy contributes no audit events and requires no target hook.

## Methods to benchmark in later experiments

### Filtering and near-duplicate detection

| Method | Use and measurement | Source |
| --- | --- | --- |
| JSONPath projection | Select nested fields from large tool responses. Check required paths, sparse records, and citation retention. | [RFC 9535](https://www.rfc-editor.org/rfc/rfc9535.html) |
| DOM main-content extraction | Extract readable article content before retrieval. Measure retention on pages with evidence in tables, sidebars, and footnotes. | [Mozilla Readability](https://github.com/mozilla/readability) |
| MinHash similarity | Estimate set overlap between shingled passages. Evaluate duplicate recall and preservation of small factual edits. | [Broder’s original paper](https://doi.org/10.1109/SEQUEN.1997.666900) |
| SimHash | Group near-duplicate blocks using compact fingerprints. Check whether an identifier or negation is enough to separate different records. | [Charikar’s original paper](https://dl.acm.org/doi/10.1145/509907.509965) |

### Retrieval and ranking

| Method | Use and measurement | Source |
| --- | --- | --- |
| BM25 | Rank keyword-rich documents with term frequency and document-length normalization. Measure evidence recall at the same prompt budget. | [Robertson and Zaragoza](https://doi.org/10.1561/1500000019) |
| Dense Passage Retrieval | Embed queries and passages and rank by vector similarity. Compare recall for paraphrases and exact identifiers. | [DPR paper](https://aclanthology.org/2020.emnlp-main.550/) |
| SPLADE | Learn sparse lexical expansion for retrieval. Measure retrieval recall alongside index size and latency. | [SPLADE paper](https://arxiv.org/abs/2107.05720) |
| HNSW | Use an approximate nearest-neighbor index for dense retrieval. Measure search latency and recall at fixed *k* before looking for prompt savings. | [HNSW paper](https://arxiv.org/abs/1603.09320) |
| Reciprocal Rank Fusion | Combine independently ranked lists using reciprocal ranks. Check whether mixing sparse and dense rankings makes a smaller shortlist feasible. | [RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) |
| Cross-encoder reranking | Score each query–passage pair jointly before final truncation. Include reranking runtime and cost in the attempt totals. | [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) |
| ColBERT late interaction | Compare individual query and passage token embeddings during ranking. Count index cost and check whether better ranking lowers the required *k*. | [ColBERT paper](https://arxiv.org/abs/2004.12832) |
| Maximal Marginal Relevance | Select passages using relevance and a redundancy penalty. Evaluate multi-document coverage and exact evidence recall. | [Carbonell and Goldstein](https://www.cs.cmu.edu/afs/cs/Web/People/jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf) |
| Listwise LLM reranking | Ask a model to order a list of passages. Include reranker input/output tokens, latency, retries, and malformed rankings. | [RankGPT paper](https://arxiv.org/abs/2304.09542) |
| Adaptive retrieval | Retrieve on demand and evaluate evidence with a learned control signal. Compare skipped retrieval against misses on retrieval-dependent cases. | [Self-RAG paper](https://arxiv.org/abs/2310.11511) |

### Prompt compression and history

| Method | Use and measurement | Source |
| --- | --- | --- |
| TextRank extraction | Rank sentences by graph centrality and retain selected sentences verbatim. Test rare evidence and ordering. | [TextRank paper](https://aclanthology.org/W04-3252/) |
| Selective Context | Keep text units with higher model-estimated information content. Test small qualifiers, negation, and exact values. | [Selective Context paper](https://arxiv.org/abs/2310.06201) |
| LLMLingua | Allocate a token budget across prompt components and compress in stages. Include the compressor and verify protected spans. | [LLMLingua paper](https://arxiv.org/abs/2310.05736) |
| LongLLMLingua | Compress long context with query-aware selection and document ordering. Test citations and questions requiring several documents. | [LongLLMLingua paper](https://arxiv.org/abs/2310.06839) |
| LLMLingua-2 | Select tokens using a distilled classifier. Account for the encoder and test token retention across domains. | [LLMLingua-2 paper](https://arxiv.org/abs/2403.12968) |
| RECOMP | Insert an extractive or abstractive compression stage after retrieval. Measure the compressor overhead and faithfulness of each form. | [RECOMP paper](https://arxiv.org/abs/2310.04408) |
| Rolling summary | Compact completed dialogue when context reaches a threshold. Grade old constraints, contradictory updates, and repeated summary drift. | [Claude context engineering cookbook](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools) |
| Schema-validated state snapshot | Keep task state, decisions, and source references in a typed object. Grade state reconstruction after compaction. | [JSON Schema 2020-12](https://json-schema.org/draft/2020-12) |
| Semantic memory retrieval | Retrieve earlier memories using relevance, recency, or importance. Test conflicting preferences, expiring facts, and missed memory retrieval. | [Generative Agents paper](https://arxiv.org/abs/2304.03442) |
| Tiered memory | Move less frequently needed context to external storage and retrieve it as needed. Count the additional memory calls and test recall on long sessions. | [MemGPT paper](https://arxiv.org/abs/2310.08560) |

### Reuse, routing, and prompt optimization

| Method | Use and measurement | Source |
| --- | --- | --- |
| Provider prompt caching | Reuse a stable input prefix across calls. Measure cold and warm bill, latency, and full input tokens separately. | [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [Claude prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| Inference-server prefix caching | Reuse key/value computation for shared prefixes. Measure prefill latency and hit rate with the same input text. | [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching.html) |
| Exact response cache | Reuse an answer for an exact key that includes the model, settings, permissions, and data version. Evaluate invalidation and request isolation. | [GPTCache project](https://github.com/zilliztech/GPTCache) |
| Semantic response cache | Reuse an answer when embedding similarity clears a threshold. Test near-miss entities, numbers, permissions, and stale answers. | [GPTCache project](https://github.com/zilliztech/GPTCache) |
| Model cascade | Attempt a cheaper model first and escalate based on a decision rule. Meter every attempted model call. | [FrugalGPT paper](https://arxiv.org/abs/2305.05176) |
| Learned model routing | Select a model from estimates of task difficulty and strengths. Hold the routing threshold fixed for evaluation and include router overhead. | [RouteLLM paper](https://arxiv.org/abs/2406.18665) |
| Offline prompt compilation | Search instructions or demonstrations on training tasks, then freeze the selected prompt. Include search cost separately and evaluate on a held-out split. | [DSPy paper](https://arxiv.org/abs/2310.03714) |
