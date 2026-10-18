# OmniFlow — Architecture

This document is the canonical reference for OmniFlow's internal design. It
covers (1) the orchestration layer, (2) the control-token protocol, (3) the
cross-modal memory, (4) ThinkOmni's fusion math, (5) Foresight's
dual-stream shared-encoder design, (6) the expert registry pattern, and (7)
the overhead analysis methodology used in the benchmark harness.

The contract surface described here is locked by the unit tests in
[`tests/`](../tests/). When the parallel implementation tasks (om-A, om-B)
ship the real modules under `src/omnimodal/`, they must conform to these
signatures exactly; the test suite's inline fallback mocks exist precisely
to lock the contract before the real implementation lands.

---

## Table of Contents

1. [Orchestration Layer Design](#1-orchestration-layer-design)
2. [Control-Token Protocol](#2-control-token-protocol)
3. [Cross-Modal Memory Design](#3-cross-modal-memory-design)
4. [ThinkOmni Fusion Math](#4-thinkomni-fusion-math)
5. [Foresight Dual-Stream with Shared Weights](#5-foresight-dual-stream-with-shared-weights)
6. [Expert Registry Pattern](#6-expert-registry-pattern)
7. [Overhead Analysis Methodology](#7-overhead-analysis-methodology)
8. [Ethics & Responsible Use](#8-ethics--responsible-use)

---

## 1. Orchestration Layer Design

The orchestration layer is a four-stage pipeline:

```
        ┌─────────┐     ┌────────────┐     ┌──────────────┐     ┌──────────┐
query ─▶│ Plan    │ ─▶ │ Resolve    │ ─▶ │ Group        │ ─▶ │ Execute  │ ─▶ answer
        │ gener-  │     │ deps       │     │ parallel     │     │ (walk    │
        │ ation   │     │ (toposort) │     │ (greedy batch)│    │ tokens)  │
        └─────────┘     └────────────┘     └──────────────┘     └──────────┘
            ▲                                                          │
            │                                                          ▼
        ┌───┴────┐                                                ┌──────────┐
        │ LLM or │                                                │  memory  │
        │ fallb. │                                                │ (writes) │
        └────────┘                                                └──────────┘
```

### 1.1 Plan generation

`LLMController.plan(query, available_experts=None) -> Plan` is the entry
point. When an LLM backend is configured (`llm_backend is not None`), the
controller prompts the LLM with:

* The user query (string).
* A JSON-schema description of the available experts.
* An instruction to emit a JSON array of `ControlToken` objects.

The controller parses the LLM's JSON output into a `Plan(tokens,
strategy="llm", n_groups=1)`. When no LLM is available, the controller
falls back to the heuristic planner (`fallback_planner`), which uses a
static keyword→expert table to choose experts and a fixed dependency
ruleset to wire them up.

### 1.2 Dependency resolution

`LLMController.resolve_dependencies(tokens) -> list[ControlToken]` performs
a deterministic topological sort. Two sources of dependency are merged:

* The `depends_on` field explicitly listed on each `ControlToken` (LLM
  declared dependencies).
* Static dependency rules defined on the controller:
  * `layout` requires `ocr`
  * `chart` requires `ocr` + `layout`
  * `vqa` requires `ocr` + `caption`
  * `reason` requires `ocr` + `layout` + `caption`

The merge uses a depth-first visit with an `on_stack` set so cycles are
detected: if a cycle is encountered the offending edge is dropped silently,
guaranteeing the resolver terminates. After resolution, each token's
`depends_on` field is rewritten to the merged set (intersected with the
set of experts actually present in the plan) so downstream consumers
(e.g., `group_parallel`) see the final dependency graph.

### 1.3 Parallel grouping

`LLMController.group_parallel(tokens) -> list[list[ControlToken]]` is a
greedy batcher. For each token in topological order:

* Compute its earliest admissible group as `1 + max(placed[dep] for dep in
  depends_on)`, or `0` if no dependencies have been placed yet.
* Extend `groups` to that index (appending empty lists as needed).
* Append the token to `groups[earliest]` and record
  `placed[token.expert] = earliest`.

After grouping, each `ControlToken.parallel_group` is set to its batch
index. The pipeline can then run each batch in parallel; the test suite
verifies this property via `test_parallel_vs_sequential_grouping`.

### 1.4 Execution

`OrchestrationPipeline.run(query, modalities=None) -> PipelineResult`
walks the resolved plan in topological order (which is the same order
produced by `group_parallel` flattened). For each token:

1. Look up the expert by `tok.expert` in `pipeline.experts`.
2. Build an `ExpertTask` carrying:
   * The token's inputs.
   * The stringified output of every upstream expert (so the chart
     expert can read the OCR'd text without an explicit graph-rewriting
     pass).
3. Call `expert.run(task) -> ExpertResult`.
4. Write the result to `CrossModalMemory` keyed by expert id + run
   ordinal, and bump `pipeline.expert_call_count`.

After all tokens execute, the pipeline picks the answer from the
highest-priority present expert (`reason` > `vqa` > `caption`), and
returns a `PipelineResult` with `plan`, `results`, `memory_writes`,
`overhead_ms`, and `expert_call_count`.

### 1.5 Fallback planner

When the LLM is unavailable (no API key, no network), the controller
routes through `fallback_planner(query, available_experts=None) -> Plan`.
The heuristic planner uses a static keyword→expert table:

| Keyword(s)        | Expert(s) triggered              |
|-------------------|----------------------------------|
| `ocr`, `read`     | `ocr`                            |
| `layout`          | `layout`                         |
| `chart`, `graph`  | `chart`                          |
| `describe`, `caption` | `caption`                    |
| `question`, `ask` | `vqa`                            |
| `reason`, `summarize` | `reason`                    |
| `document`        | `ocr`, `layout`, `chart`, `vqa`  |

If no keyword matches, the planner falls back to `[caption, reason]`. The
heuristic planner is **deterministic** given the same input text — the
same seed produces the same plan byte-for-byte (verified by
`test_pipeline_is_deterministic_across_runs`).

---

## 2. Control-Token Protocol

### 2.1 Token structure

```python
@dataclass
class ControlToken:
    expert: str                   # canonical expert id, e.g. "ocr"
    modality: Modality            # TEXT | IMAGE | DOCUMENT | VIDEO | AUDIO | TABLE | CHART
    inputs: list[str]             # input ports (raw queries or upstream expert ids)
    depends_on: list[str]         # expert ids that must run before this one
    parallel_group: int | None    # filled in by group_parallel()
    params: dict[str, Any]        # backend-specific knobs (dpi, lang, top_k, ...)

    @property
    def id(self) -> str:
        return f"{self.expert}:{':'.join(self.inputs)}"
```

### 2.2 Why control tokens (not function calling)?

The control-token protocol is intentionally simpler than the OpenAI /
Anthropic function-calling protocol. Three reasons:

1. **Backend-agnostic.** Control tokens are JSON; any LLM that can be
   prompted to emit JSON can be an OmniFlow controller.
2. **Inspectable.** A `Plan` is a plain list of `ControlToken`s that can
   be serialised, diffed, and replayed.
3. **Resolvable.** Because tokens carry explicit dependencies, the
   resolver can merge them with static rules and topologically sort
   without re-prompting the LLM.

### 2.3 Plan structure

```python
@dataclass
class Plan:
    tokens: list[ControlToken]    # the resolved, sorted tokens
    strategy: str                 # "llm" | "heuristic"
    parallel_groups: int          # number of parallel batches
```

### 2.4 Serialization

`Plan` is a Pydantic dataclass so it serialises to JSON via
`plan.model_dump_json()` (or `json.dumps(plan, default=vars)` when the
inline mock is used). This makes plans first-class artefacts that can be
logged, replayed, and benchmarked.

---

## 3. Cross-Modal Memory Design

`CrossModalMemory` is a deliberately small, in-process scratchpad. Its
design goals are:

* **Single-run lifetime.** The memory lives for one `pipeline.run()` call
  (or one user session if you choose to share it). It is **not** a
  long-horizon memory system; for that, swap in a vector DB.
* **Modality-aware.** Every stored item carries a `Modality` tag so
  retrieval can be filtered (e.g., "give me back only the image
  captions").
* **Embedding-optional.** When no embedding backend is supplied, the
  memory falls back to a deterministic **Jaccard-over-token-sets**
  similarity metric. This guarantees reproducibility in the offline
  test environment.

### 3.1 MemoryItem

```python
@dataclass
class MemoryItem:
    id: str
    modality: Modality
    content: Any                  # typically a string or dict
    embedding: list[float] | None # optional, used if cosine similarity is configured
    timestamp: float
    metadata: dict[str, Any]
```

### 3.2 Retrieval algorithm

```
retrieve(query, top_k, modality=None) -> list[MemoryItem]
  1. filter by modality (if provided)
  2. score each remaining item by similarity(query, item.content)
  3. stable-sort by score desc, then by id asc (deterministic tiebreak)
  4. return the first top_k items
```

### 3.3 Similarity metrics

| Metric      | Formula                                                          | When to use                          |
|-------------|------------------------------------------------------------------|--------------------------------------|
| Jaccard     | $\frac{|A \cap B|}{|A \cup B|}$ over whitespace token sets       | Default; offline, deterministic      |
| Cosine      | $\frac{\vec{e}_a \cdot \vec{e}_b}{\|\vec{e}_a\| \cdot \|\vec{e}_b\|}$ | When an embedding backend is supplied|

The default is Jaccard because it has no RNG dependence and produces
byte-identical scores across runs and machines. Cosine is the upgrade
path: supply a `MemoryItem.embedding` field and switch the
`similarity` config to `cosine`.

### 3.4 Capacity and eviction

The memory is FIFO-bounded at `capacity` (default 1024). When the
capacity is exceeded, the oldest item is evicted. This is intentional:
for a single pipeline run, recent context matters more than distant
context, and FIFO is the cheapest eviction policy that preserves
ordering.

### 3.5 Context building

`build_context(query, top_k=5) -> str` concatenates the retrieved items
with a `" | "` separator, producing a single string suitable for prompt
injection. This is the bridge between the memory and the LLM's
context window.

---

## 4. ThinkOmni Fusion Math

### 4.1 Notation

Let:

* $V$ be the shared vocabulary of size $|V|$.
* $p_\text{LLM} \in \Delta^{V-1}$ be the LLM's next-token distribution
  (after softmax).
* $p_\text{perc} \in \Delta^{V-1}$ be the perceptual expert's
  distribution over the same vocabulary.
* $\alpha \in [0, 1]$ be the LLM weight.
* $\tau$ be the conflict threshold (default 0.3).

### 4.2 Weighted contrastive scaling

The fused logits (in probability space) are

$$\ell_\text{fused}(v) = \alpha \cdot p_\text{LLM}(v) - (1 - \alpha) \cdot p_\text{perc}(v).$$

The fused distribution is

$$p_\text{fused}(v) = \text{softmax}\!\left(\ell_\text{fused}(v)\right).$$

The subtraction (rather than addition) is deliberate: when the perceptual
side confidently supports token $v$, the LLM's prior on $v$ should be
*punished* when the LLM was indifferent to it, not amplified. This is
the **contrastive** in the strategy name.

### 4.3 Adaptive alpha

The weight is computed adaptively from the KL divergence between the two
distributions:

$$\alpha = \alpha_\text{min} + \sigma\!\left(\mathrm{KL}(p_\text{perc} \,\|\, p_\text{LLM})\right) \cdot (\alpha_\text{max} - \alpha_\text{min}),$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic sigmoid.

Intuition: when the perceptual distribution diverges sharply from the
LLM (high KL), we trust the LLM more (it has stronger language priors).
When they agree (low KL), we balance them evenly ($\alpha \to
\alpha_\text{min}$). The squashing via sigmoid keeps $\alpha$ bounded in
$[\alpha_\text{min}, \alpha_\text{max}]$ even when the KL is very large.

### 4.4 Conflict detection

A conflict is flagged iff:

$$\arg\max_v p_\text{LLM}(v) \neq \arg\max_v p_\text{perc}(v)
\quad \text{and} \quad
\mathrm{JSD}(p_\text{LLM} \,\|\, p_\text{perc}) > \tau,$$

where the Jensen-Shannon divergence is

$$\mathrm{JSD}(P \| Q) = \frac{1}{2} \mathrm{KL}(P \| M) + \frac{1}{2} \mathrm{KL}(Q \| M), \quad M = \frac{P + Q}{2}.$$

The JSD is bounded in $[0, \ln 2]$, so $\tau = 0.3$ is a reasonable
default that fires on genuinely diverging distributions.

When a conflict is flagged, a `ConflictSignal(type, severity, source,
target, llm_top, perc_top)` is attached to the `decode()` output dict.
The `severity` field is the JSD itself, so consumers can rank conflicts
across a sequence.

### 4.5 Fusion strategies

| Strategy                | Formula                                                        | Notes                                     |
|-------------------------|----------------------------------------------------------------|-------------------------------------------|
| `weighted_contrastive` | $\alpha \cdot p_\text{LLM} - (1-\alpha) \cdot p_\text{perc}$   | Default; sharpens disagreements.          |
| `weighted_average`     | $\alpha \cdot p_\text{LLM} + (1-\alpha) \cdot p_\text{perc}$   | Smooth ensembling; no contrastive edge.   |
| `geometric_mean`       | $\sqrt{p_\text{LLM} \cdot p_\text{perc}}$                     | Robust to outliers; both sides must agree.|
| `max_entropy`          | picks the side with higher $H(p)$                             | Trust the more-uncertain side.            |

### 4.6 Why probability-space fusion (not logit-space)?

Real LLMs expose their next-token distribution only after softmax;
expert backends (BLIP, ViLT, OCR) likewise expose scores that are
already normalised. Fusing in probability space makes heterogeneous
backends directly comparable without a calibration step. The trade-off
is that we lose the magnitude information from raw logits; for very
low-entropy experts the contrastive strategy can produce a tied-fused
argmax, in which case the LLM wins by tiebreak.

---

## 5. Foresight Dual-Stream with Shared Weights

### 5.1 Why dual-stream?

Single-stream video-LLMs face a fundamental trade-off:

* **Process every frame.** Smooth, but expensive (every frame runs the
  full encoder + decoder).
* **Process every Nth frame.** Cheap, but jumpy — the model has no
  representation of the in-between frames and cannot answer questions
  about them.

Foresight resolves the trade-off by running **two streams over a single
shared encoder backbone**:

* The **now** stream processes every `frame_skip`-th frame at high
  temporal resolution.
* The **future** stream peeks ahead at the in-between frames (those
  whose indices are not multiples of `frame_skip`), reusing the now
  stream's encoder activations as KV-cache priors.

### 5.2 Shared encoder

```
                ┌───────────────────────────────────────────┐
                │              Shared encoder                │
                │  (single backbone, single set of weights)  │
                └──────┬──────────────────────────┬──────────┘
                       │                          │
                       ▼                          ▼
                ┌─────────────┐           ┌──────────────┐
                │  now stream │           │ future stream │
                │  (frames N, │           │ (frames N+1,  │
                │  N+skip, …) │           │  N+skip+1, …) │
                └─────────────┘           └──────────────┘
                       │                          │
                       └──────────┬───────────────┘
                                  ▼
                          merged predictions
```

The encoder is **the same Python object** for both streams — `now_stream.encoder
is future_stream.encoder` returns `True` (verified by
`test_dual_stream_shares_encoder`). This is the contract-level guarantee
that the two streams never disagree on the underlying representation.

### 5.3 Frame processing

`process_frame(frame) -> dict` runs the now stream on a single frame.
It returns:

```python
{
  "index": int,
  "prediction": str,
  "stream": "now",
  "shared_encoder": bool,
  "interpolated": False,
  "frame_count": int,
}
```

### 5.4 Future planning

`plan_future(frames, lookahead=None) -> list[Frame]` selects up to
`lookahead` frames whose indices are not multiples of `frame_skip`. This
is the future stream's work queue. The selection is purely
index-based (so a dropped frame still appears in the future plan);
drop-handling is the consumer's responsibility at execution time.

### 5.5 Frame-drop handling

When a frame is marked `dropped=True`, `handle_dropped_frame(frame)`
synthesises a prediction from the last successful one:

* If a prior prediction exists, its `prediction` string is wrapped as
  `f"interpolated:{last_prediction}"` and the new dict is returned with
  `interpolated=True`.
* If no prior prediction exists (the very first frame is dropped), a
  neutral stub `"interpolated:neutral"` is returned.

This guarantees that downstream consumers always see a continuous
signal; the `interpolated` flag lets them choose to re-run the affected
window later.

### 5.6 Configuration

| Knob             | Default | Meaning                                                   |
|------------------|---------|-----------------------------------------------------------|
| `frame_skip`     | 2       | now stream processes every Nth frame                      |
| `lookahead`      | 3       | future stream peeks this many frames ahead                |
| `dual_stream`    | True    | set False to disable the future stream entirely           |
| `shared_encoder` | None   | if None, a default shared encoder is created lazily      |

### 5.7 Production path (future work)

In the mock layer the "shared encoder" is a single Python object
instance shared between two streams. In production with a real
transformer backbone, the production path is:

1. Allocate **one** transformer encoder (e.g., the visual encoder of a
   Video-LLaMA-class model).
2. Maintain a single KV cache populated by the now stream's recent
   frames.
3. When the future stream processes an in-between frame, run it through
   the same encoder but with the now stream's KV cache as a
   cross-attention prior (rather than re-encoding from scratch).

This plumbing is left as future work; the contract surface in this
release is the structural guarantee (`encoder_shared=True`).

---

## 6. Expert Registry Pattern

Every expert conforms to the `BaseExpert` protocol:

```python
class BaseExpert(Protocol):
    name: str
    modality: Modality
    latency_ms: float
    call_count: int

    def run(self, task: ExpertTask) -> ExpertResult: ...
```

### 6.1 Backend injection

Each expert accepts a `backend` object at construction:

```python
ocr = OCRExpert(backend=easyocr.Reader(["en"]))
```

The expert calls `backend.predict(...)` (or whatever the backend's
canonical method is) inside its `_produce(task)` method. The pipeline
never talks to the backend directly; it only ever calls
`expert.run(task)`. This makes backends swappable without touching the
pipeline.

### 6.2 Mock backend

When no backend is supplied, each expert falls back to a deterministic
mock that returns canned output keyed off the task's inputs. The mock is
seeded so the same input produces the same output across runs (verified
by `test_same_input_produces_same_output_across_instances`).

### 6.3 Latency tracking

Every `run()` call measures wall time via `time.perf_counter()` and
records `latency_ms = max(elapsed_ms, configured_floor)`. The floor
(`latency_ms` constructor kwarg) is the configured minimum latency for
the expert — it lets the benchmark harness report consistent numbers in
mock mode even when the wall time is dominated by Python overhead.

### 6.4 Registry

The pipeline's `experts: dict[str, BaseExpert]` is the registry. Add
experts via `pipeline.register_expert(name, expert)`. The controller's
`plan(available_experts=...)` argument lets you restrict which experts
are eligible for a given query.

### 6.5 Error handling

`BaseExpert.run()` wraps `_produce()` in a try/except. On exception,
the result's `success=False` flag is set and the exception text is
returned as `output`. This lets the pipeline continue executing
downstream experts even when one fails (degraded mode).

---

## 7. Overhead Analysis Methodology

The benchmark harness measures three quantities per pipeline run:

1. **Wall time** — `time.perf_counter()` from before `plan()` to after
   the last expert returns.
2. **Per-expert latency** — summed `ExpertResult.latency_ms` across all
   executed tokens.
3. **Overhead** — `overhead_ms = max(0, wall_ms - sum(expert.latency_ms))`.

The overhead metric is the time spent in the framework itself:
controller planning, dependency resolution, parallel grouping, memory
writes, result aggregation. It excludes the time spent inside experts.
A well-tuned implementation should keep overhead below ~10% of wall
time for typical document-understanding workflows.

### 7.1 Methodology caveats

* `time.perf_counter()` resolution is platform-dependent; on Linux it
  is ~1 ns, on Windows it can be ~10 ns. The 0.5 s test suite runtime
  hides this for our purposes.
* Python's GIL means truly parallel expert execution in a single process
  is impossible; the parallel-grouping pass produces *correct* grouping
  but the executor walks batches sequentially. True parallelism is on
  the 0.3.0 roadmap (`asyncio.gather` executor).
* The mock experts have no I/O and no real work; their `latency_ms` is
  the configured floor. Real backends will see very different
  overhead/wall ratios.

### 7.2 Reproducibility

The harness is fully seeded (`--seed 42` is the default). Every run
produces the same plan structure, the same expert call counts, and the
same wall time modulo OS jitter. Results are written to
`results/benchmark_<timestamp>.json` with the full plan, per-expert
latencies, and the computed overhead.

---

## 8. Ethics & Responsible Use

OmniFlow is a framework, not a product. The ethical weight lives in
the experts you plug in. Please:

### 8.1 Vet expert backends

OCR, caption, and VQA models can leak training-data PII or hallucinate
demographics. Audit each backend's training data and known failure
modes before deploying OmniFlow in any user-facing context.

### 8.2 Surface conflicts

ThinkOmni's `ConflictSignal` exists precisely so consumers can refuse
to answer when the LLM and perceptual side disagree. Do not silently
suppress conflict signals in production; route them to a repair expert
or to a calibrated low-confidence fallback.

### 8.3 Rate-limit and log

The bundled FastAPI service has no built-in rate limiting or audit log.
Deploy behind a gateway that does both. Log every `/orchestrate` and
`/stream` request with the full plan, expert outputs, and conflict
signals so you can reconstruct decisions after the fact.

### 8.4 Do not use for surveillance

Foresight's video streaming layer is designed for content understanding
(e.g., educational video QA, meeting summarisation). It is **not**
designed for tracking individuals. If your use case drifts toward
surveillance, stop and consult your ethics board — the framework's
dual-stream architecture is intentionally optimised for content
understanding, not for re-identification.

### 8.5 Bias audits

The fallback heuristic planner uses a static keyword→expert table that
reflects the authors' defaults. Downstream tasks with non-English
queries or non-Western document layouts may need a re-tuned table. Run
a bias audit before deploying the heuristic planner in production.

### 8.6 Reporting

When you publish results obtained with OmniFlow, report:

* The expert backends used (with versions).
* The LLM backend used (with model identifier).
* The full `Plan` for at least one representative query.
* The conflict rate (% of decode steps that flagged a conflict).
* The overhead/wall ratio.

This transparency lets the community reproduce and stress-test your
numbers.

---

*This document is the canonical architecture reference. Pull requests
that change the documented contract surface must update this file in the
same PR.*
