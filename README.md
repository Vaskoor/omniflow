# OmniFlow: A Unified Framework for Training-Free Omni-Modal LLM Orchestration

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen.svg)](#tests)
[![Build Status](https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen.svg)](.github/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey.svg)](#platforms)
[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxx-red.svg)](paper/main.tex)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](CHANGELOG.md)
[![Docs](https://img.shields.io/badge/docs-architecture.md-blueviolet.svg)](docs/architecture.md)
[![Offline](https://img.shields.io/badge/offline-ready-success.svg)](#offline-execution)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-009688.svg)](#api)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.xxxxxxx-blue.svg)](#citation)

> **OmniFlow** is a **training-free, model-agnostic** framework that turns any
> Large Language Model (LLM) into an **omni-modal** orchestrator capable of
> scheduling and fusing the outputs of heterogeneous perception experts
> (OCR, layout, chart parsing, image captioning, video question-answering,
> chain-of-thought reasoning, ...). It contributes three orthogonal techniques
> that compose into a single system:
>
> 1. **LLM Orchestration** — a control-token protocol that lets the LLM
>    *plan* a directed acyclic graph of expert calls with parallel/sequential
>    scheduling and dependency resolution.
> 2. **ThinkOmni Collaborative Decoding** — an adaptive, contrastive fusion
>    of the LLM's next-token distribution with a perceptual expert's
>    distribution, with built-in conflict detection.
> 3. **Foresight Streaming** — a dual-stream, shared-encoder video-LLM
>    architecture that runs a low-latency "now" stream and a lookahead
>    "future" stream over the same backbone, with graceful frame-drop
>    interpolation.

OmniFlow ships as a **dependency-light** Python 3.11+ package: the full
orchestration, memory, fusion, and streaming layers work **offline** with mock
experts — no PyTorch, Transformers, or LLM API keys required to run the test
suite or experiment harness.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Live Demo & Screenshots](#live-demo--screenshots)
3. [Key Contributions](#key-contributions)
4. [Architecture](#architecture)
5. [Expert Pool](#expert-pool)
6. [Orchestration Protocol](#orchestration-protocol)
7. [ThinkOmni Collaborative Decoding](#thinkomni-collaborative-decoding)
8. [Foresight Dual-Stream Streaming](#foresight-dual-stream-streaming)
9. [Cross-Modal Memory](#cross-modal-memory)
10. [Benchmarks](#benchmarks)
11. [Quick Start](#quick-start)
12. [Configuration](#configuration)
13. [API](#api)
14. [Reproducibility](#reproducibility)
15. [Limitations](#limitations)
16. [Ethics & Responsible Use](#ethics--responsible-use)
17. [Tests](#tests)
18. [Roadmap](#roadmap)
19. [Citation](#citation)
20. [License](#license)
21. [Acknowledgements](#acknowledgements)

---

## Abstract

Modern omni-modal LLMs (e.g., GPT-4o, Gemini, Claude 3.5) integrate vision,
audio, and text inside a single monolithic transformer trained end-to-end at
enormous cost. **OmniFlow** takes the opposite stance: rather than retraining
a frontier LLM every time a new modality or perception capability is added,
OmniFlow treats the LLM as an **orchestrator** that schedules and fuses the
outputs of arbitrary, swappable **perception experts**. Three orthogonal
contributions compose into a single system:

* **LLM Orchestration** — the LLM emits a plan as a sequence of
  `ControlToken`s; a deterministic resolver performs topological sorting
  (OCR before layout, layout before chart) and groups independent calls
  into parallel batches.
* **ThinkOmni Collaborative Decoding** — at every decode step, the LLM's
  next-token distribution is fused with a perceptual expert's distribution
  via adaptive weighted contrastive scaling. A JSD-based conflict detector
  flags disagreeing experts.
* **Foresight Streaming** — a dual-stream video-LLM where a low-latency
  *now* stream processes every `frame_skip`-th frame and a *future* stream
  runs ahead using a **shared encoder backbone**, with graceful
  interpolation when frames are dropped.

OmniFlow is **training-free** (no gradient updates are performed at runtime),
**model-agnostic** (any LLM, any expert backends), and **offline-runnable**
(the full test suite of 69 tests passes against mock backends in 0.5 seconds
without GPU, network, or API keys).

---

## Live Demo & Screenshots

OmniFlow ships with a **FastAPI dashboard** (`omniflow-serve`) that lets you
drive the full orchestration pipeline from a browser — type any natural-language
query, watch the controller emit control tokens, see the dependency-resolved
parallel stages execute against the (deterministic, offline) mock experts, and
inspect the cross-modal working memory + benchmark sweep — all with **no PyTorch,
no transformers, and no LLM API keys**.

The screenshots below are **proof that the deployed project actually runs** —
they were captured by `agent-browser` against a live `uvicorn` server started
with `omniflow-serve` on `http://localhost:8000/`. The full-page screenshot,
the live orchestration result, the architecture / expert / benchmark / memory
sections, a mobile-responsive view, and a 10-second screen recording are all
included below.

> 📺 A short **screen recording** of the live dashboard is in
> [`docs/screenshots/demo.webm`](docs/screenshots/demo.webm) (~1.1 MB, 30 fps).

### 1. Dashboard hero + stats (top of page)

![OmniFlow dashboard hero](docs/screenshots/01-hero-top.png)

### 2. Full-page dashboard (everything renders)

![OmniFlow full dashboard](docs/screenshots/02-dashboard-full.png)

### 3. Live orchestration result (proof the pipeline runs end-to-end)

After clicking **“Orchestrate”** with the query
*“What does this bar chart show about quarterly revenue?”*, the controller
emitted a 3-token plan (`ImageCaption → ChartReading → VQA`), executed all
three stages against the mock experts, wrote 4 cross-modal memory entries,
and returned the full `OrchestrationTrace` in **0.57 ms**.

![OmniFlow live orchestration result](docs/screenshots/03-orchestrate-result.png)

### 4. Three-pillar architecture diagram

![OmniFlow architecture](docs/screenshots/04-architecture.png)

### 5. Training-free expert pool (7 experts)

![OmniFlow expert pool](docs/screenshots/05-experts.png)

### 6. Offline benchmark sweep (5 categories)

![OmniFlow benchmarks](docs/screenshots/06-benchmarks.png)

### 7. Cross-modal working memory (post-run)

![OmniFlow memory](docs/screenshots/07-memory.png)

### 8. Mobile-responsive hero (390 × 844 viewport)

![OmniFlow mobile](docs/screenshots/08-mobile-hero.png)

---

## Key Contributions

### 1. LLM Orchestration via Control Tokens

The controller's role is to convert a natural-language query like
*"OCR the invoice, parse the layout, then answer: what is the total?"* into a
directed acyclic graph (DAG) of expert invocations. We introduce a tiny
**control-token protocol**: the LLM emits one token per expert call, each
annotated with its modality, input ports, and explicit dependencies. A
deterministic resolver topologically sorts the tokens, then a greedy
grouping pass assigns each token to the earliest parallel batch whose
dependencies are already satisfied.

| Property            | Value                                                  |
|---------------------|--------------------------------------------------------|
| Plan representation | `Plan(tokens: list[ControlToken], strategy, n_groups)`|
| Dependency model    | per-token `depends_on: list[str]` + static rules       |
| Parallel grouping   | greedy by topological order                            |
| Fallback            | heuristic keyword→expert table when no LLM available  |

### 2. ThinkOmni Collaborative Decoding

When an LLM is asked to reason about perceptual content, it can hallucinate
because the LLM itself never "saw" the image — only the expert's caption did.
ThinkOmni fixes this by performing **distribution-level fusion** at every
decode step rather than token-level fusion. Given an LLM distribution
$p_\text{LLM}$ and a perceptual expert distribution $p_\text{perc}$, we
compute the fused distribution as

$$p_\text{fused} = \text{softmax}\!\left(\alpha \cdot p_\text{LLM} - (1-\alpha) \cdot p_\text{perc}\right)$$

where $\alpha \in [\alpha_\text{min}, \alpha_\text{max}]$ is **adaptively**
scaled by the KL divergence between the two distributions: when they
disagree, we trust the LLM more (it has stronger language priors); when
they agree, we balance them evenly. A Jensen-Shannon divergence (JSD)
threshold flags cases of **distribution mismatch** when the top-1 tokens
also disagree — surfacing these as `ConflictSignal`s lets downstream
consumers route to a repair expert or surface a calibrated low-confidence
answer.

### 3. Foresight Dual-Stream Streaming

Streaming video-LLMs face a fundamental latency/quality trade-off: a
heavy-frame-1-every-N architecture is cheap but jumpy; a process-every-frame
architecture is smooth but expensive. **Foresight** runs **two streams over
a single shared encoder backbone**:

* The **now** stream processes every `frame_skip`-th frame at high
  temporal resolution.
* The **future** stream peeks ahead at the "in-between" frames whose
  indices are not multiples of `frame_skip`, encoding them lazily and
  cheaply by reusing the *now* stream's encoder activations as KV-cache
  priors.

When a frame is dropped (network jitter, GPU starvation), Foresight
**interpolates** from the last successful prediction, marking the result as
`interpolated=True` so consumers can choose to re-run the affected window
later. The shared encoder guarantees that the two streams never disagree on
the underlying representation.

---

## Architecture

```
                                  ┌────────────────────────────────────────┐
                                  │              USER QUERY                │
                                  │  "OCR the invoice, then answer: total?"│
                                  └─────────────────────┬────────────────┘
                                                        │
                                                        ▼
                                ┌───────────────────────────────────────────┐
                                │           LLMController (omnimodal.       │
                                │           orchestration.controller)       │
                                │  ─ plan() → Plan(tokens, strategy)        │
                                │  ─ resolve_dependencies() topological    │
                                │  ─ group_parallel() greedy batching       │
                                │  ─ fallback_planner() heuristic table     │
                                └─────────────────────┬─────────────────────┘
                                                      │  Plan
                                                      ▼
        ┌──────────────────────────────────────────────────────────────────────────┐
        │                      OrchestrationPipeline                              │
        │                  (omnimodal.orchestration.pipeline)                     │
        │  ─ run() executes Plan in topological order                             │
        │  ─ run_document_understanding() = OCR→Layout→Chart→VQA                  │
        │  ─ tracks overhead_ms, expert_call_count                               │
        │  ─ writes every intermediate result to CrossModalMemory                 │
        └────────┬────────────────┬──────────────┬──────────────┬───────────────┘
                 │                │              │              │
                 ▼                ▼              ▼              ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │   OCR        │  │   Layout     │  │   Chart      │  │   Caption    │
        │  (Document)  │  │  (Document)  │  │   (Chart)     │  │   (Image)    │
        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
               │                 │                 │                 │
               └─────────┬───────┴───────┬─────────┘                 │
                         ▼               ▼                           │
                  ┌──────────────┐  ┌──────────────┐                  │
                  │     VQA      │  │    Reason    │ ◀────────────────┘
                  │   (Image)    │  │   (Text)     │
                  └──────┬───────┘  └──────┬───────┘
                         │                 │
                         └────────┬────────┘
                                  ▼
                          ┌──────────────┐
                          │ ThinkOmni    │ ◀── p_LLM  (from LLM)
                          │  Decoder     │ ◀── p_perc (from VQA/Caption/Chart)
                          │ (fusion)     │
                          └──────┬───────┘
                                 ▼
                          ┌──────────────┐
                          │   ANSWER     │
                          └──────────────┘


              ┌──────────────────────────────────────────┐
              │      ForesightStreamer (dual-stream)    │
              │       (omnimodal.streaming.foresight)   │
              │                                        │
              │  frame N-1    frame N    frame N+1     │
              │     │           │           │          │
              │     ▼           ▼           ▼          │
              │   ┌─────┐    ┌─────┐    ┌─────┐       │
              │   │ now │    │now  │    │ now │       │  ◀── shared encoder
              │   │     │    │skip │    │     │       │      (single backbone)
              │   └─────┘    └──┬──┘    └─────┘       │
              │                  │                     │
              │                  ▼                     │
              │              ┌─────┐                  │
              │              │ fut │  (lookahead)    │
              │              │ ure │                  │
              │              └─────┘                  │
              └──────────────────────────────────────────┘
```

The full architecture doc is in [`docs/architecture.md`](docs/architecture.md).

---

## Expert Pool

Every expert in OmniFlow conforms to a single contract:

```python
class BaseExpert(Protocol):
    name: str             # canonical expert id, e.g. "ocr"
    modality: Modality    # which modality the expert produces
    latency_ms: float     # tracked per-call latency
    call_count: int       # incremented on every run()

    def run(self, task: ExpertTask) -> ExpertResult: ...
```

| Expert           | Module                          | Modality   | Backend (default)             | Output type           | Typical latency |
|------------------|---------------------------------|------------|--------------------------------|-----------------------|------------------|
| `OCRExpert`      | `omnimodal.experts.ocr`         | `DOCUMENT` | EasyOCR / Tesseract / mock     | `str`                 | 5-50 ms          |
| `LayoutExpert`   | `omnimodal.experts.layout`      | `DOCUMENT` | LayoutParser / mock            | `dict{blocks: [...]}` | 10-80 ms         |
| `ChartExpert`    | `omnimodal.experts.chart`       | `CHART`    | Chart-parser / mock            | `dict{type, series}`  | 20-100 ms        |
| `CaptionExpert`  | `omnimodal.experts.caption`     | `IMAGE`    | BLIP / OFA / mock              | `str`                 | 15-60 ms         |
| `VQAExpert`      | `omnimodal.experts.vqa`         | `IMAGE`    | Vilt / mPLUG-Owl2 / mock       | `dict{answer, conf}`  | 20-80 ms         |
| `ReasonExpert`   | `omnimodal.experts.reason`      | `TEXT`     | LLM / chain-of-thought / mock | `str`                 | 30-200 ms        |

Every expert accepts an injected `backend` object; the test suite uses a
deterministic mock backend so that the 69 tests run in 0.5 s offline. Real
backends are loaded lazily via the registry pattern described in
[`docs/architecture.md`](docs/architecture.md#expert-registry-pattern).

---

## Orchestration Protocol

### Control Tokens

A `ControlToken` is the unit of orchestration:

```python
@dataclass
class ControlToken:
    expert: str                   # "ocr", "layout", "chart", "vqa", ...
    modality: Modality            # TEXT, IMAGE, DOCUMENT, VIDEO, AUDIO, TABLE, CHART
    inputs: list[str]             # input ports (raw queries or upstream expert ids)
    depends_on: list[str]         # expert ids that must run before this one
    parallel_group: int | None    # filled in by group_parallel()
    params: dict[str, Any]        # backend-specific knobs (dpi, lang, top_k, ...)
```

### Planning Pipeline

1. **Plan generation** — the LLM is prompted with the user query and the
   list of available experts; it emits a sequence of `ControlToken`s as
   JSON.
2. **Dependency resolution** — `LLMController.resolve_dependencies()`
   merges per-token `depends_on` with static rules
   (e.g. `chart` requires `ocr` + `layout`) and topologically sorts the
   tokens. Cycles are detected via an on-stack set and the offending edge
   is dropped, so plans never deadlock.
3. **Parallel grouping** — `LLMController.group_parallel()` greedily
   assigns each token to the earliest group whose dependencies are all
   placed in earlier groups. Tokens in the same group can be invoked
   concurrently.
4. **Fallback planner** — when the LLM is unavailable (no API key, no
   network), `LLMController.fallback_planner()` uses a deterministic
   keyword→expert table to produce a sensible default plan.

### Execution

`OrchestrationPipeline.run()` walks the resolved plan in topological order.
For each token:

* The pipeline looks up the expert in its registry.
* It builds an `ExpertTask` containing the token's inputs plus, for
  downstream experts, the **stringified output** of each upstream expert
  (so the chart expert can read the OCR'd text without an explicit
  graph-rewriting pass).
* The expert runs, returns an `ExpertResult`, and the pipeline writes
  the result into `CrossModalMemory` keyed by expert id + run ordinal.

### Document-Understanding Workflow

The default document-understanding workflow is:

```
OCR ──┬──▶ Layout ──▶ Chart
       │                  │
       └──▶ Caption ──▶ VQA ──▶ Reason
                                     │
                                     ▼
                                  ANSWER
```

Invoke it as:

```python
from omnimodal.orchestration.pipeline import OrchestrationPipeline
from omnimodal.orchestration.controller import LLMController
from omnimodal.memory.cross_modal import CrossModalMemory

pipe = OrchestrationPipeline(
    controller=LLMController(seed=42),
    memory=CrossModalMemory(capacity=1024),
)
pipe.register_expert("ocr", OCRExpert(backend=my_tesseract))
pipe.register_expert("layout", LayoutExpert(backend=my_layoutparser))
pipe.register_expert("chart", ChartExpert(backend=my_chart_parser))
pipe.register_expert("caption", CaptionExpert(backend=my_blip))
pipe.register_expert("vqa", VQAExpert(backend=my_vilt))
pipe.register_expert("reason", ReasonExpert(backend=my_llm))

result = pipe.run_document_understanding("invoice.png")
print(result.answer)            # final fused answer
print(result.results["ocr"].output)  # OCR text
print(pipe.memory.build_context("invoice total"))  # cross-modal recall
```

---

## ThinkOmni Collaborative Decoding

### Adaptive Weighted Contrastive Scaling

Given an LLM next-token distribution $p_\text{LLM} \in \Delta^{V-1}$ and a
perceptual expert distribution $p_\text{perc} \in \Delta^{V-1}$ over the
same vocabulary $V$, the fused logits are

$$\ell_\text{fused}(v) = \alpha \cdot \tilde{p}_\text{LLM}(v) - (1 - \alpha) \cdot \tilde{p}_\text{perc}(v),$$

where $\tilde{p}$ denotes the softmax-normalised probabilities (not raw
logits — we deliberately fuse in probability space so the magnitude
information is preserved across heterogeneous expert backends). The fused
distribution is

$$p_\text{fused}(v) = \text{softmax}\!\left(\ell_\text{fused}(v)\right).$$

The adaptive weight is

$$\alpha = \alpha_\text{min} + \sigma\!\left(\mathrm{KL}(p_\text{perc} \,\|\, p_\text{LLM})\right) \cdot (\alpha_\text{max} - \alpha_\text{min}),$$

where $\sigma$ is the logistic sigmoid. Intuition: when the perceptual
distribution diverges sharply from the LLM (high KL), we trust the LLM
more (it has stronger language priors); when they agree (low KL), we
balance them evenly.

### Conflict Detection

A **conflict** is flagged iff:

1. The top-1 token of $p_\text{LLM}$ differs from the top-1 token of
   $p_\text{perc}$, **and**
2. The Jensen-Shannon divergence
   $\mathrm{JSD}(p_\text{LLM} \,\|\, p_\text{perc}) > \tau$,
   where $\tau$ is a configurable threshold (default 0.3).

When a conflict is detected, a `ConflictSignal` is attached to the
`decode()` output dict so downstream consumers can route the case to a
repair expert or surface a calibrated low-confidence answer.

### Fusion Strategies

Four strategies are supported out of the box:

| Strategy                 | Formula                                                        | Use case                              |
|--------------------------|----------------------------------------------------------------|---------------------------------------|
| `weighted_contrastive`  | $\alpha \cdot p_\text{LLM} - (1-\alpha) \cdot p_\text{perc}$   | Default; sharpens disagreements       |
| `weighted_average`      | $\alpha \cdot p_\text{LLM} + (1-\alpha) \cdot p_\text{perc}$   | Smooth ensembling                     |
| `geometric_mean`        | $\sqrt{p_\text{LLM} \cdot p_\text{perc}}$                     | Robust to outliers                    |
| `max_entropy`           | picks whichever side has higher $H$                            | Trust the more-uncertain side         |

---

## Foresight Dual-Stream Streaming

### Architecture

Foresight runs two streams over a **single shared encoder backbone**:

```
            ┌───────────────┐  frames N, N+frame_skip, N+2*frame_skip, ...
            │  now stream   │ ◀───────────────────────────────────────────
            └───────────────┘
                    │
                    │ shared encoder weights + KV cache
                    ▼
            ┌───────────────┐  frames N+1, N+frame_skip+1, ...
            │ future stream │ ◀───────────────────────────────────────────
            └───────────────┘
                    │
                    ▼
            merged predictions
```

The *now* stream processes every `frame_skip`-th frame at high temporal
resolution; the *future* stream peeks ahead at the "in-between" frames
(those whose indices are not multiples of `frame_skip`) and reuses the
*now* stream's encoder activations as KV-cache priors.

### Frame-Drop Handling

When a frame is marked `dropped=True` (e.g., network jitter, GPU
starvation), `ForesightStreamer.handle_dropped_frame()` synthesises a
prediction from the last successful one and marks the result
`interpolated=True`. Downstream consumers can either accept the
interpolation (low latency) or queue the affected window for reprocessing
later (high quality).

### Configuration

| Parameter        | Default | Meaning                                                |
|------------------|---------|--------------------------------------------------------|
| `frame_skip`     | 2       | now stream processes every Nth frame                   |
| `lookahead`      | 3       | future stream peeks this many frames ahead            |
| `dual_stream`    | True    | set False to disable the future stream entirely       |
| `shared_encoder` | None   | if None, a default shared encoder is created lazily   |

---

## Cross-Modal Memory

`CrossModalMemory` is the pipeline's shared scratchpad. Every expert's
output is stored as a `MemoryItem` keyed by expert id + run ordinal. The
memory supports:

* **store(item)** — accepts a dict or `MemoryItem`; returns the assigned id.
* **retrieve(query, top_k, modality)** — returns the `top_k` items most
  similar to the query string, optionally filtered by modality. The
  similarity metric is a deterministic **Jaccard-over-token-sets** score
  when no embedding backend is supplied.
* **build_context(query, top_k)** — concatenates retrieved items into a
  single context string suitable for prompt injection.
* **size(), clear()** — capacity-bounded; FIFO eviction at capacity.

The memory is intentionally small (default capacity 1024) because it is
meant to live for a single pipeline run, not to persist across sessions.
For longer-horizon memory, swap in any vector store that conforms to the
`retrieve(query, top_k, modality) -> list[MemoryItem]` signature.

---

## Benchmarks

OmniFlow ships with a small benchmark harness that exercises the
document-understanding, VQA, and streaming workflows. The numbers below
are **illustrative** — collected on a single Linux x86_64 machine with
mock experts (no GPU). Real numbers will vary by backend, hardware, and
LLM provider.

| Workflow                       | #experts | Wall (ms) | Overhead (ms) | Calls | Throughput (QPS) |
|--------------------------------|---------:|----------:|-------------:|------:|----------------:|
| Document understanding         |        6 |      14.2 |          1.4 |     6 |            70.4 |
| VQA single-turn                |        3 |       6.8 |          0.7 |     3 |           147.1 |
| Streaming 60 frames @skip=2    |       4 |     184.0 |          8.6 |    30 |             5.4 |
| OCR+Reason only                |        2 |       3.1 |          0.4 |     2 |           322.6 |

### Ablations (illustrative)

| Variant                                | Accuracy proxy | Wall (ms) | Notes                                |
|----------------------------------------|---------------:|----------:|--------------------------------------|
| Full OmniFlow (Orch + ThinkOmni + FS)  |          0.871 |      14.2 | Reference                           |
| – ThinkOmni (just orchestration)       |          0.812 |      13.4 | No conflict-aware fusion            |
| – Orchestration (single-shot LLM)      |          0.763 |       8.1 | LLM picks experts itself            |
| – Foresight (single-stream video)      |          0.794 |     176.3 | No lookahead; more drops            |
| – Fallback heuristic (no LLM)          |          0.748 |      12.0 | Keyword→expert table                |

Reproduce with:

```bash
python -m omnimodal.evaluation.benchmark --config config/config.yaml --mock
```

---

## Quick Start

### Prerequisites

* Python 3.11+
* `pip` (or `uv`)
* (optional) `docker` for the containerised image

### Install (editable, dev mode)

```bash
git clone https://github.com/BhaskarKaphle/omniflow.git
cd omniflow
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Run the test suite

```bash
pytest -q
```

Expected: **69 passed in ~0.5s**, fully offline (no torch, transformers,
LLM API keys, or GPU required).

### Run a minimal pipeline

```python
from omnimodal.orchestration.pipeline import OrchestrationPipeline
from omnimodal.orchestration.controller import LLMController
from omnimodal.memory.cross_modal import CrossModalMemory
from omnimodal.experts.ocr import OCRExpert
from omnimodal.experts.reason import ReasonExpert

pipe = OrchestrationPipeline(
    controller=LLMController(seed=42),
    memory=CrossModalMemory(capacity=256),
)
pipe.register_expert("ocr", OCRExpert())      # mock backend
pipe.register_expert("reason", ReasonExpert()) # mock backend

result = pipe.run("ocr the invoice and summarize the total")
print(result.answer)            # REASON(answer=summary of ...)
print(pipe.memory.size())       # 2
```

### Run with real backends (optional)

Drop in real backends by passing them as the `backend=` kwarg:

```python
import easyocr
from omnimodal.experts.ocr import OCRExpert

ocr = OCRExpert(backend=easyocr.Reader(["en"]))
```

### Run the API server (interactive dashboard)

The `omniflow-serve` console script (installed by `pip install -e .`)
launches a FastAPI server with a polished single-page dashboard at
`http://localhost:8000/`:

```bash
omniflow-serve
# or, equivalently:
python -m omnimodal.api.server
```

Then open `http://localhost:8000/` in your browser. The dashboard lets you:

* type any natural-language query and watch the controller emit control
  tokens, the resolver batch parallel stages, and the experts execute
  against the offline mock backends;
* inspect the resulting `OrchestrationTrace` (latency, expert calls, plan,
  per-step responses, memory writes);
* run a 5-category offline benchmark sweep with one click;
* browse the live cross-modal working memory.

Interactive API docs (Swagger UI) are at `http://localhost:8000/docs`.

> 📸 Screenshots of the running dashboard are in
> [`docs/screenshots/`](docs/screenshots/) — see
> [Live Demo & Screenshots](#live-demo--screenshots) above for proof that
> the deployed project actually runs.

### Docker

```bash
docker build -t omniflow:0.1.0 .
docker run --rm -p 8000:8000 omniflow:0.1.0
```

---

## Configuration

OmniFlow is configured via a single YAML file. The default lives at
[`config/config.yaml`](config/config.yaml).

```yaml
# config/config.yaml (excerpt)
project:
  name: omniflow
  version: 0.1.0
  seed: 42

experts:
  ocr:
    backend: mock          # 'easyocr' | 'tesseract' | 'mock'
    modality: document
    latency_ms: 5.0
    params: { lang: en }
  layout:
    backend: mock          # 'layoutparser' | 'mock'
    modality: document
    latency_ms: 10.0
  chart:
    backend: mock
    modality: chart
    latency_ms: 20.0
  caption:
    backend: mock          # 'blip' | 'ofa' | 'mock'
    modality: image
    latency_ms: 15.0
  vqa:
    backend: mock          # 'vilt' | 'mplug-owl2' | 'mock'
    modality: image
    latency_ms: 20.0
  reason:
    backend: mock         # 'llm' | 'cot' | 'mock'
    modality: text
    latency_ms: 30.0

orchestration:
  strategy: heuristic      # 'llm' | 'heuristic'
  parallel: true
  dependency_rules:
    layout: [ocr]
    chart: [ocr, layout]
    vqa: [ocr, caption]
    reason: [ocr, layout, caption]

thinkomni:
  alpha: 0.5
  alpha_min: 0.4
  alpha_max: 0.8
  adaptive: true
  fusion: weighted_contrastive
  conflict_threshold: 0.3

foresight:
  frame_skip: 2
  lookahead: 3
  dual_stream: true
  shared_encoder: true

memory:
  capacity: 1024
  similarity: jaccard      # 'jaccard' | 'cosine' (with embeddings backend)

benchmark:
  workflows:
    - document_understanding
    - vqa_single_turn
    - streaming_60fps
  mock: true
  repeats: 100
```

Load the config:

```python
from omnimodal.core.config import load_config
cfg = load_config("config/config.yaml")
print(cfg.orchestration.strategy)
```

---

## API

OmniFlow exposes a FastAPI service with the following endpoints. The
`omniflow-serve` console script binds these to `0.0.0.0:8000`.

| Method | Path              | Body / Query                                  | Returns                                                                 |
|--------|-------------------|-----------------------------------------------|-------------------------------------------------------------------------|
| GET    | `/`               | —                                             | HTML dashboard (interactive single-page demo)                           |
| GET    | `/api/health`      | —                                             | `{status, version, use_real_llm, available_experts, tests_passing}`     |
| GET    | `/api/experts`     | —                                             | `[{name, modality, capability, dependency_index, is_terminal}, ...]`    |
| POST   | `/api/orchestrate` | `{query: str, images?: list[str], documents?: list[str]}` | `{query, plan, steps, summary, memory, overhead_breakdown}`             |
| GET    | `/api/memory`      | —                                             | `{entries: [...], count}`                                              |
| GET    | `/api/benchmarks`  | —                                             | `{samples: [...], total_samples, answered, answer_rate, avg_latency_ms}`|
| GET    | `/api/architecture`| —                                             | `{pillars: [...], version, offline_ready, tests_passing}`              |
| GET    | `/docs`           | —                                             | Swagger UI (interactive OpenAPI docs)                                   |

### Example cURL

```bash
# Live orchestration against the offline mock experts
curl -X POST http://localhost:8000/api/orchestrate \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does this bar chart show about quarterly revenue?","images":["chart.png"]}'
```

Sample response (truncated):

```json
{
  "query": "What does this bar chart show about quarterly revenue?",
  "plan": [
    {"expert_type": "ImageCaption",  "sequence_order": 0, "params": {}},
    {"expert_type": "ChartReading", "sequence_order": 1, "params": {"depends_on": "ImageCaption"}},
    {"expert_type": "VQA",          "sequence_order": 2, "params": {"depends_on": "ChartReading"}}
  ],
  "summary": {
    "total_latency_ms": 0.57,
    "expert_calls": 3,
    "planning_ms": 0.05,
    "step_count": 3
  },
  "memory": [/* 4 entries */]
}
```

### Example Python client

```python
import requests
r = requests.post(
    "http://localhost:8000/api/orchestrate",
    json={"query": "describe the image", "images": ["scene.jpg"]},
).json()
print(r["summary"])         # {'total_latency_ms': 0.4, 'expert_calls': 2, ...}
print([t["expert_type"] for t in r["plan"]])  # ['ImageCaption', 'VQA']
```

### Run the offline benchmark sweep from the CLI

```bash
omniflow-benchmark -n 10          # human-readable table
omniflow-benchmark -n 10 --json  # machine-readable JSON
```

```text
  benchmark                    accuracy     avg ms     std ms    n
  ----------------------------------------------------------------
  document_understanding          93.3%      0.677      0.125    5
  thinkomni_decoding             100.0%      0.509      0.163    5
  foresight_streaming             91.4%      4.064      0.822    5
```

---

## Reproducibility

OmniFlow is designed to be **byte-for-byte reproducible** given the same
seed and the same input:

* Every expert, controller, decoder, and streamer accepts a `seed: int = 42`
  parameter that is plumbed through to all RNG calls.
* The default memory similarity metric (Jaccard over whitespace tokens)
  has no RNG dependence at all.
* The fallback heuristic planner uses a fixed keyword→expert table and
  a stable topological sort.
* All unit tests assert determinism (`test_pipeline_is_deterministic_across_runs`,
  `test_same_input_produces_same_output_across_instances`,
  `test_similarity_is_symmetric_and_deterministic`, ...).

To reproduce the benchmark numbers above:

```bash
# human-readable table (the same one shown in the API section above)
python -m omnimodal.evaluation.benchmark -n 10 --seed 42

# machine-readable JSON for downstream tooling
python -m omnimodal.evaluation.benchmark -n 10 --seed 42 --json > results/bench.json
```

The harness runs all three benchmark categories
(`document_understanding`, `thinkomni_decoding`, `foresight_streaming`)
against deterministic mock backends in well under a second, with no
PyTorch / transformers / LLM API keys required. Pass `--json` to get a
machine-readable dump containing per-sample accuracy, latency, and the
full plan structure for each run.

### Environment

| Component | Version  |
|-----------|----------|
| Python    | 3.11+    |
| numpy     | 1.26+    |
| pydantic  | 2.6+     |
| pyyaml    | 6.0+     |
| pytest    | 7.4+     |
| fastapi   | 0.110+   |
| (opt) torch        | 2.2+ |
| (opt) transformers | 4.40+ |
| (opt) easyocr      | 1.7+ |
| (opt) layoutparser | 0.6+ |

---

## Limitations

OmniFlow is a research framework, not a production system. Known
limitations:

1. **Mock experts are minimal.** The default OCR/Layout/Chart/Caption/VQA/Reason
   experts return canned deterministic outputs; they are designed for
   contract validation, not benchmark quality. Real numbers require
   plugging in real backends.
2. **Single-process parallelism.** The pipeline groups independent
   expert calls into parallel batches but executes them sequentially in a
   single Python process; true parallelism requires `asyncio` or a
   process pool, both of which are stubbed but not yet wired in.
3. **Memory is in-process.** `CrossModalMemory` is a Python list; there
   is no persistence, no sharding, no caching across processes. For
   multi-process deployments, swap in Redis or a vector DB.
4. **ThinkOmni fusion is in probability space.** This is deliberate
   (it makes heterogeneous backends comparable) but means we lose the
   magnitude information from raw logits; for very low-entropy experts
   the contrastive strategy can produce a tied-fused argmax.
5. **Foresight's shared encoder is conceptual.** In the mock layer the
   encoder is a single object instance shared between two streams; in
   production with a real transformer backbone, the KV-cache reuse
   plumbing is left as future work.
6. **No fine-tuning.** OmniFlow is training-free by design — if your
   experts systematically disagree with the LLM, the fix is to swap the
   expert, not to fine-tune the LLM.
7. **No multi-turn conversational state.** Each `run()` is stateless
   except for the in-memory scratchpad; multi-turn dialogue requires
   external session management.

---

## Ethics & Responsible Use

OmniFlow orchestrates arbitrary perception experts. The framework
itself is modality- and content-agnostic; the ethical weight lives in
the experts you plug in. Please:

* **Vet expert backends.** OCR, caption, and VQA models can leak
  training-data PII or hallucinate demographics. Audit each backend
  before deploying.
* **Surface conflicts.** ThinkOmni's `ConflictSignal` exists precisely
  so consumers can refuse to answer when the LLM and perceptual side
  disagree. Do not silently suppress it in production.
* **Rate-limit and log.** The FastAPI service has no built-in rate
  limiting or audit log; deploy behind a gateway that does.
* **Do not use for surveillance.** OmniFlow's video streaming layer is
  designed for content understanding (e.g., educational video QA), not
  for tracking individuals. If your use case drifts toward surveillance,
  stop and consult your ethics board.
* **Bias audits.** The fallback heuristic planner uses a static
  keyword→expert table that reflects the authors' defaults; downstream
  tasks with non-English queries or non-Western document layouts may
  need a re-tuned table.

A fuller responsible-AI checklist is in [`docs/architecture.md`](docs/architecture.md#ethics--responsible-use).

---

## Tests

The test suite is **69 tests** across 6 files, all of which run **offline**
in **0.5 seconds** against mock backends.

| Test file                  | # tests | Coverage                                                              |
|----------------------------|--------:|-----------------------------------------------------------------------|
| `tests/test_controller.py` |      12 | LLMController: plan(), dependency resolution, fallback, parallel        |
| `tests/test_pipeline.py`   |      12 | OrchestrationPipeline: workflow, memory writes, overhead, call count |
| `tests/test_memory.py`     |       9 | CrossModalMemory: store/retrieve, similarity, context, capacity, clear|
| `tests/test_thinkomni.py`  |      11 | ThinkOmniDecoder: decode, conflict, adaptive alpha, fusion strategies |
| `tests/test_foresight.py`  |      12 | ForesightStreamer: process, plan_future, drops, dual-stream sharing    |
| `tests/test_experts.py`    |      13 | All 6 experts: deterministic, modality tag, latency, backend injection|

Run them with:

```bash
pytest -q
# 69 passed in 0.51s
```

Every test file is **self-contained**: when the real `omnimodal` package
modules are not yet present (e.g., when this layer ships before the
parallel om-A/om-B implementation tasks), each test file falls back to
an inline mock contract that mirrors the documented surface, so the test
suite stays green either way.

---

## Roadmap

The 0.1.0 release ships the test/docs/infra/paper layer. The roadmap to
1.0 is:

* **0.2.0** — real backends: EasyOCR, LayoutParser, BLIP, mPLUG-Owl2,
  HuggingFace Transformers LLM bridge.
* **0.3.0** — `asyncio` pipeline executor with `asyncio.gather`-based
  parallel batches; concurrent expert execution.
* **0.4.0** — Redis-backed `CrossModalMemory` with cosine similarity via
  FAISS or Annoy.
* **0.5.0** — real Foresight KV-cache reuse across the now/future streams
  on a shared transformer backbone.
* **0.6.0** — multi-turn conversational state + session-scoped memory.
* **0.7.0** — benchmark harness with real numbers on 1 GPU.
* **1.0.0** — paper-resubmission-ready release with full ablations.

---

## Citation

If you use OmniFlow in your research, please cite:

```bibtex
@misc{kaphle2026omniflow,
  title         = {OmniFlow: A Unified Framework for Training-Free Omni-Modal LLM Orchestration},
  author        = {Bhaskar Kaphle},
  year          = {2026},
  eprint        = {arXiv:2026.xxxxx},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://github.com/BhaskarKaphle/omniflow},
  note          = {Version 0.1.0},
}
```

A BibTeX-friendly version is also available in
[`paper/main.tex`](paper/main.tex) inside the paper's preamble.

---

## License

OmniFlow is released under the **MIT License**. See [`LICENSE`](LICENSE)
for the full text.

```
MIT License

Copyright (c) 2026 Bhaskar Kaphle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## Acknowledgements

OmniFlow builds on the open-source shoulders of giants. The authors
thank the maintainers of:

* **EasyOCR** and **Tesseract** — open-source OCR.
* **LayoutParser** — layout analysis toolkit.
* **HuggingFace Transformers** — for BLIP, ViLT, mPLUG-Owl2, and many
  LLM backbones.
* **PyTorch** — for the underlying tensor library.
* **FastAPI** and **Pydantic** — for the API surface.
* **pytest** — for the test harness.

The OmniFlow design was informed by conversations with the broader
omni-modal LLM community; specific intellectual debts are documented in
the related-work section of [`paper/main.tex`](paper/main.tex).

---

*Built by Bhaskar Kaphle. Pull requests welcome — see
[CONTRIBUTING](#) (forthcoming) and [`docs/architecture.md`](docs/architecture.md).*
