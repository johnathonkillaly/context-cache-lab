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
what is published (§5) separately from the small set of things this repo actually adds
(§6) and the genuine remaining gaps (§7). Stages 1–4 are best understood as *replicating
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

```bash
.venv/bin/python scripts/run_stage2a.py --lengths 4096,16384 --limit-docs 4
```

```bash
.venv/bin/python scripts/smoke_stage2b.py --steps 200 --docs 2
```

```bash
.venv/bin/python scripts/train_stage2b.py --steps 2200 --lr 5e-5
```

```bash
.venv/bin/python scripts/run_stage2b_eval.py --limit-docs 8
```

```bash
.venv/bin/python scripts/make_tables_s2b.py
```

```bash
.venv/bin/python scripts/measure_stage2b_cost.py
```

`make_tables.py` exits non-zero when the Stage 1 gate fails, so the gate is a check, not
a judgement call made in prose afterwards.

## Layout

| Path | Contents |
|---|---|
| [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) | Protocol and **pre-registered, frozen** success criteria |
| [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | Literature survey; published vs. ours |
| [`docs/STAGE2B_DESIGN.md`](docs/STAGE2B_DESIGN.md) | Sidecar design + the eight training-objective questions |
| [`docs/FUTURE_IDEAS.md`](docs/FUTURE_IDEAS.md) | Deliberately deferred ideas + their preconditions |
| [`AGENTS.md`](AGENTS.md) | Status, failures, results, next steps (Claude Code / Codex) |
| `src/ccl/` | Library code |
| `scripts/` | Entry points |
| `tests/` | Automated tests |
| `results/` | Raw JSON/CSV + generated tables |

## Status

See [`AGENTS.md`](AGENTS.md) for the full record. Short version:

- **Stage 0 complete** — literature survey written before any code.
- **Stage 1 complete, gate PASSED** — `NATIVE` 0.947 vs `NOCTX` 0.000 over 720
  evaluations, with no parametric leakage on any exact fact class. Native prefill at 16K
  costs **22.7 s** (24.7 s TTFT, 2.4 GiB of KV) on this M4 Max — the cost that motivates
  the whole project.
- **Stage 2a complete** — training-free compression floor. Two findings that shape
  everything after it:
  - **Dropping KV positions is catastrophic even at 2×** (0.944 → 0.028). A learned
    compressor is *necessary*, not an optimization.
  - **Equal-budget raw text scores ≈ 1/r** (0.556 / 0.250 / 0.139 / 0.056 at
    2/4/8/16×) and beats every KV heuristic at every ratio. So the real question for a
    learned compressor is sharp: **does a degraded version of every chunk beat a perfect
    version of some chunks?**
- **Stage 2b complete — gate verdict INCONCLUSIVE** (not failed). A 566M-parameter
  C²KV-style sidecar (14% of the target, target frozen) trained 2,200 steps.
  - **The primary question got a yes:** approximate-everything beats exact-some at every
    ratio, and the margin *widens* as budget shrinks — LEARNED vs BUDGET is
    0.569/0.472 at 2×, 0.389/0.222 at 4×, 0.347/0.125 at 8×, 0.278/0.069 at 16×.
  - **But two of four pre-registered PROMISING criteria missed by ~1%** (LEARNED@4×
    0.389 vs 0.40; semantic rank margin +1.66 vs +2.0). **Thresholds were not moved.**
  - Controls clean: RANDOM 0.014, WRONGPAGE 0.069 vs LEARNED 0.389 — no answer leakage.
  - **Independent compilation is not worse than joint** — and is *better* at 8× and 16×.
  - **Warm TTFT is 23–32× faster than native prefill; cold is slower.** Breakeven at
    **2 queries** against the same document.
  - **Semantics survive, exact strings do not**: `semantic` 0.625 at 4× while `hash`
    collapses to 0.000 and `identifier` to 0.250 (where raw BUDGET wins at 0.750).
- **Stages 3–8 not started.**

## Relationship to `epitaxy`

Separate experiment (cross-model KV transfer). This repo does **not** import its code or
mix its measurements. At most it borrows an evaluation idea conceptually, and says so.
