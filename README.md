# context-cache-lab

**Can long LLM context be compiled once into reusable, compact, model-readable state —
instead of forcing the target model to repeatedly prefill the original tokens?**

A research repo. The goal is to learn whether this architecture deserves another
experiment, not to produce an impressive number.

---

## The question

Take a document split into chunks `C_1 … C_n`. Compile each chunk **independently** into
a compact state `Z_i`, with no knowledge of the other chunks. Later, compose them:

```
target( [Z_1; Z_2; …; Z_k] + q )   ≟   target( [C_1; C_2; …; C_k] + q )
```

If that holds — and if it holds without native prefill over the original tokens — then
context becomes a **cacheable, composable, content-addressable asset** rather than
something re-derived on every request.

## The mental model: virtual memory, not RAG

```
L1 — active model context      tiny set of selected compiled pages
L2 — compressed page states    precompiled, reusable
L3 — exact raw-token pages     lossless
L4 — original document         storage
```

Storage is cheap. Repeated target-model attention over every original token is
expensive. So keep information losslessly *outside* the active attention set, and pay
for exact context only when the query actually needs it.

The long-term structure is a content-addressed page:

```
P_i = (h_i, r_i, z_i, C_i)
      │    │    │    └── exact original tokens (lossless fallback)
      │    │    └─────── compiled model-readable state
      │    └──────────── tiny retrieval landmark
      └───────────────── stable content ID (SHA-256 — an ID only, NOT an embedding)
```

## Honesty first

**This idea is largely not novel.** [C²KV (KDD 2026)](https://arxiv.org/abs/2607.17715)
already does independently-compiled, composable, compressed KV with a frozen base model.
[Cartridges (2025)](https://arxiv.org/abs/2506.06266) already established the
compile-once/amortize economics. [Landmark Attention
(2023)](https://arxiv.org/abs/2305.16300) already does block-level retrieval through
attention.

[`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) was written **before any code**, and lists
what is published (§4) separately from the small set of things this repo actually adds
(§5) and the genuine remaining gaps (§6). Stages 1–4 are best understood as *replicating
C²KV's core claim on Apple Silicon with tighter cost accounting.*

## Method commitments

Taken from the literature, fixed before running anything:

- **The target model stays frozen.** Only a lightweight sidecar is trained.
- **Store `Z` pre-RoPE; re-apply positions at composition time.** Letting every chunk
  believe it sits at position 0 is an *ablation*, not the design.
- **KV-shaped carrier, not last-layer embeddings** — retains detail far better at ratio.
- **Start at 2–8× compression.** Not 256×.
- **Cold cost is never hidden.** Every speedup is reported as cold TTFT, warm TTFT, and
  a full amortization curve over m ∈ {1,2,4,8,16,32} queries.

## Controls

Every quality number ships with: `NATIVE`, `NOCTX`, `JOINT`, `RANDOM`, `SHUFFLE`,
`WRONGPAGE`, and `BUDGET` (raw context truncated to the same token count as `Z`).

`BUDGET` is the one that matters most and the one this literature most often omits: a
compiled state that beats no-context but loses to the same number of raw tokens has
demonstrated nothing.

## Hardware

Apple M4 Max, 128 GB unified memory, macOS. PyTorch 2.14 / MPS. No CUDA assumptions, no
FlashAttention CUDA kernels. Target model `Qwen/Qwen3-4B` (already cached locally — the
repo does not download large checkpoints without checking the cache first).

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

## Commands

```bash
.venv/bin/python -m pytest -m "not slow"
```

```bash
.venv/bin/python scripts/probe_model.py --model Qwen/Qwen3-4B
```

```bash
.venv/bin/python scripts/build_corpus.py --draw A --lengths 1024,2048,4096,8192,16384
```

```bash
.venv/bin/python scripts/run_native_baseline.py --draw A --model Qwen/Qwen3-4B
```

```bash
.venv/bin/python scripts/make_tables.py --results results/raw/stage1_baseline_drawA.json
```

`make_tables.py` exits non-zero when the Stage 1 gate fails, so the gate is a check, not
a judgement call made in prose afterwards.

## Layout

| Path | Contents |
|---|---|
| [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) | Protocol and **pre-registered, frozen** success criteria |
| [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | Literature survey; published vs. ours |
| [`docs/FUTURE_IDEAS.md`](docs/FUTURE_IDEAS.md) | Deliberately deferred ideas + their preconditions |
| [`AGENTS.md`](AGENTS.md) | Status, failures, results, next steps (Claude Code / Codex) |
| `src/ccl/` | Library code |
| `scripts/` | Entry points |
| `tests/` | Automated tests |
| `results/` | Raw JSON/CSV + generated tables |

## Status

See [`AGENTS.md`](AGENTS.md). Short version: Stage 0 complete, Stage 1 in progress.

## Relationship to `epitaxy`

Separate experiment (cross-model KV transfer). This repo does **not** import its code or
mix its measurements. At most it borrows an evaluation idea conceptually, and says so.
