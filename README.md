# context-cache-lab

**Can long LLM context be compiled once into compact, independently generated, composable
state that a frozen model reads instead of the original tokens?**

We built it, measured it, and **the general claim did not survive.** A learned compressed
carrier beat equal-budget raw context on single-document QA (Stage 2b), then **failed
outright** on a preregistered cross-page compositional task (Stage 3).

This repository preserves that outcome rather than burying it.

> ### Failed stages stay failed.
>
> Success criteria were frozen in writing **before** each held-out evaluation and have not
> been edited afterwards. Stage 2b missed two of four thresholds by ~1% and is recorded as
> **INCONCLUSIVE**, not rounded up. Stage 3 is recorded as **FAIL**. Neither has been
> reframed, softened, or retested until it passed. The negative result is the deliverable.

---

## Result at a glance

| Stage | Question | Verdict | Headline |
|---|---|---|---|
| **0** | Literature, protocol, preregistration | — | Core idea is **not novel**; C²KV and Cartridges cover it |
| **1** | Is the harness able to measure anything? | ✅ **PASS** | NATIVE `0.947` vs NOCTX `0.000`, zero parametric leakage |
| **2a** | Do training-free heuristics suffice? | ⚪ measurement | KV thinning collapses at **2×** (`0.944 → 0.028`); raw `BUDGET ≈ 1/r` |
| **2b** | Does *learned* compressed state beat equal-budget raw text? | 🟡 **INCONCLUSIVE** | Beat BUDGET at **every** ratio, but missed 2/4 frozen criteria by ~1% |
| **3** | Do independently compiled pages support real two-page reasoning? | ❌ **FAIL** | **0/13** valid items vs NATIVE **13/13**; all 5 gates missed |
| 2c | Bit-width compression (Axis B) | ⛔ not run | Gated on 2b passing — it did not |
| 4–8 | Reuse, pages, retrieval, fallback | ⛔ not run | Gated on Stage 3 — it failed |

**The bottom line:** compiled context state is *mechanically* sound and *economically*
attractive when it works — but this carrier does not support the composable-page
abstraction the architecture was proposed on.

---

## What this repository is

**A C²KV-style replication and measurement study on Apple Silicon**, with two
methodological tightenings the source literature does not apply:

1. **Exactness accounting.** Published compressed-memory work benchmarks semantic and
   multi-hop QA in aggregate. Here every result is broken out across nine fact classes —
   semantic, composition, proper nouns, numbers, dates, hashes, identifiers, paths, URLs —
   each with a matched, deliberately confusable distractor.
2. **Honest cold/warm accounting.** C²KV reports TTFT with offline extraction *excluded*.
   Here compilation cost is always reported, alongside warm TTFT and the query count at
   which compilation amortizes.

**What it is not:** a novel architecture, a benchmark win, or a system. No claim in this
repository is presented as new that is not new.

---

## Literature context — published vs. measured

**The central hypothesis is not ours.** Independently compiling chunks into compressed,
composable KV against a *frozen* target is exactly
[C²KV (KDD 2026)](https://arxiv.org/abs/2607.17715).
[Cartridges (2025)](https://arxiv.org/abs/2506.06266) established the compile-once /
amortize economics. [Landmark Attention (2023)](https://arxiv.org/abs/2305.16300)
established block-level retrieval through attention.
[`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) was written **before any code ran** and
separates published prior art (§5) from what this repo measured (§6) and genuine open
gaps (§7).

Design decisions taken directly from that literature, not invented here:

| Decision | Source |
|---|---|
| Freeze the target; train only a sidecar | C²KV; ICAE/500xCompressor |
| Store `Z` **pre-RoPE**, re-apply position at composition | C²KV |
| KV-shaped carrier rather than last-layer embeddings | 500xCompressor vs. ICAE |
| Supervise the answer *after* concatenation, not next-token on the corpus | C²KV; Cartridges' negative result |
| Expect trouble past ~2× for generic compression | Cartridges |

**What this repository contributes is measurement, not mechanism**: per-fact-class loss
curves, cold/warm/amortized cost, an aggressive control battery, and a preregistered
negative composition result on consumer Apple hardware.

---

## The hypothesis under test

Split a document into chunks `C₁ … Cₙ`. Compile each **independently**, with no knowledge
of the others. Later, compose:

```
target( [Z₁; Z₂; …; Z_k] + q )   ≟   target( [C₁; C₂; …; C_k] + q )
```

The intended payoff is a memory hierarchy rather than RAG:

```
L1 — active model context      tiny set of selected compiled pages
L2 — compressed page states    precompiled, reusable, content-addressed
L3 — exact raw-token pages     lossless fallback
L4 — original document         storage
```

**Stage 3 tested the load-bearing step of that picture and it did not hold.**

---

## Experimental progression

### Stage 1 — native baseline ✅ PASS

A deterministic synthetic corpus (two draws with provably disjoint value pools), 40
documents, 720 evaluations, `Qwen/Qwen3-4B` frozen on MPS.

- NATIVE **0.947** vs NOCTX **0.000** on `clean_hit` (correct *and* not fooled by the
  matched distractor).
- NOCTX rank margins sit at ≈0.00–0.13 on every exact class — the model has **no
  parametric access** to these values. This is what makes every later comparison mean
  something.
- Native prefill at 16K: **22.7 s**, TTFT **24.7 s**, 2388 MiB of KV. Throughput decays
  1100 → 747 tok/s with length, so removing active positions should pay off superlinearly.

### Stage 2a — training-free floor ⚪ measurement

Before training anything, find out what heuristics achieve.

| condition | 2× | 4× | 8× | 16× |
|---|---|---|---|---|
| KV stride-drop | 0.028 | 0.000 | 0.000 | 0.000 |
| KV sink+stride | 0.056 | 0.014 | 0.000 | 0.000 |
| **raw BUDGET** | **0.556** | **0.250** | **0.139** | **0.056** |

Two consequences that shaped everything after:

1. **Training-free KV thinning is catastrophic even at 2×** (0.944 → 0.028, rank margins
   +5.36 → ≈+0.1). A learned compressor is *necessary*, not an optimization.
2. **`BUDGET` tracks 1/r almost exactly** — the signature of an all-or-nothing baseline.
   That sharpened the real question from "beat no-context" to: **does a degraded version
   of every chunk beat a perfect version of some chunks?**

A third result is a useful piece of mechanism: shuffling *post-RoPE* KV tensors is an
**exact no-op**, because attention is permutation-invariant over keys and position lives
inside each rotated key. That is a direct demonstration of why pre-RoPE storage is
required — and why a KV-order shuffle is worthless as a composition control.

### Stage 2b — learned compressed state 🟡 INCONCLUSIVE

A 566M-parameter C²KV-style sidecar (14.1% of the target; target fully frozen), trained
2,200 steps with the compression ratio sampled per example.

**The primary question got a yes.** Approximate-everything beat exact-some at every ratio,
and the margin *widened* as budget shrank:

| ratio | LEARNED | BUDGET | Δ | active positions |
|---|---|---|---|---|
| 2× | **0.569** | 0.472 | +0.097 | 2253 |
| 4× | **0.389** | 0.222 | +0.167 | 1201 |
| 8× | **0.347** | 0.125 | +0.222 | 677 |
| 16× | **0.278** | 0.069 | +0.208 | 412 |

NATIVE 0.958, NOCTX 0.000. Controls clean: RANDOM `0.014`, WRONGPAGE `0.069` — **no answer
leakage**.

**Semantic information survived; exact strings did not.** At 4×:

| class | NATIVE | LEARNED | BUDGET | |
|---|---|---|---|---|
| semantic | 1.000 | **0.625** | 0.000 | survives, BUDGET at floor |
| composition | 0.625 | 0.375 | 0.000 | survives |
| identifier | 1.000 | 0.250 | **0.750** | **raw wins** |
| hash | 1.000 | **0.000** | 0.250 | **total collapse** |

**Warm reuse was extremely fast, and cold was not.** At 4096 tokens (native TTFT 3.219 s):

| ratio | compile | warm TTFT | warm speedup | cold TTFT | breakeven |
|---|---|---|---|---|---|
| 4× | 5.31 s | **0.114 s** | **28.2×** | 5.42 s | **m = 2** |
| 8× | 4.87 s | **0.102 s** | **31.6×** | 4.97 s | **m = 2** |

Compilation costs *more* than one native prefill, so the first query is slower. From the
**second** query onward the compiled path wins, asymptotically ~30×. This is precisely the
cost C²KV's TTFT methodology omits.

**Why INCONCLUSIVE anyway.** Two of four preregistered PROMISING criteria missed:

| criterion | threshold | measured | |
|---|---|---|---|
| LEARNED @4× | ≥ 0.40 | **0.389** | miss by 0.011 |
| semantic rank margin @4× | ≥ +2.0 | **+1.66** | miss by 0.34 |
| LEARNED − NOCTX | ≥ 0.35 | 0.389 | pass |
| JOINT within 10% | ≤ 0.10 | 0.097 | pass |

**The thresholds were not moved.** A ~1% miss is exactly the situation preregistration
exists to protect against.

### Stage 3 — cross-page composition ❌ FAIL

The frozen step-2200 carrier, at its primary 4× ratio, evaluated once on a held-out
preregistered corpus built specifically to require joining **two separate pages** —
2,358 rows across 210 variants in ~59 minutes.

**All five frozen gates missed:**

| Gate | Threshold | Result |
|---|---|---|
| 1 · retained quality | ≥ 0.60 | **0.000** |
| 2 · missing-page dependence | ≥ 0.80 | **0.077** |
| 3 · independence tax | ≥ 0.85 | non-informative (both zero) |
| 4 · scaling 2→16 pages | ≥ 0.75 | non-informative (both zero) |
| 5 · control separation | ≥ 0.15 | **0.000** |

On the 13 items where NATIVE genuinely composes (and neither page alone suffices),
NATIVE scores **13/13** and **INDEPENDENT scores 0/13** — as do JOINT, BUDGET, NOCTX,
RANDOM and WRONGPAGE. The Wilson 95% interval for independent accuracy is **0–22.8%**,
entirely below the 0.60 bar. This is not a near miss.

- **INDEPENDENT's mean rank margin (−2.47) is worse than NOCTX (−1.10).**
- **Order:** NATIVE solves all 12 ordered/shuffled pairs; INDEPENDENT and JOINT solve none.
- **JOINT fails too**, so this is *not* an independence tax and page selection would not
  rescue it — the failure appears with only two relevant pages and zero distractors.

---

## What the negative result does and does not establish

**Does:** this frozen carrier does not deliver usable compositional reasoning over
independently compiled pages on this task, at the preregistered quality level. The
proposed general composable-page claim is **not supported**.

**Does not:**

- It does **not** show all independently compiled memory architectures fail. Stage 3 used a
  new corpus whose pages are ~94–100 tokens against ~256-token training chunks, with a
  different format, task and terminal style — all changing together. This measures
  **insufficient transfer of this carrier to this task**.
- It does **not** overturn Stage 2b. Those measurements stand on their own corpus.
- It does **not** isolate an order-only defect: content recall may fail too.
- It **cannot** fully separate loss of within-page relations from failure to combine intact
  relations — the design lacks a page-local query assay on the new pages. That limitation
  is stated in [`results/stage3/stage3_report.md`](results/stage3/stage3_report.md) rather
  than glossed.

Per the preregistration, the correct response is to **stop optimizing this claim**. Any
follow-up must be a separately preregistered experiment on a fresh draw — not a retest or
relabelling of Stage 3.

---

## Reproducibility

### Environment

| | |
|---|---|
| Hardware | Apple M4 Max, 128 GB unified memory (results as reported) |
| OS | macOS (Darwin 27.0) |
| Python | 3.12 |
| PyTorch | 2.14.0, MPS backend |
| transformers | 5.16.1 |
| Target model | `Qwen/Qwen3-4B` (Apache-2.0), bf16, **frozen** |

No CUDA is assumed and no FlashAttention CUDA kernels are used. Any Apple Silicon Mac with
enough unified memory should work; ~10 GB is needed for the model plus a 16K KV cache, and
~17 GB during Stage 2b training. Non-Apple machines will fall back to CPU and be slow.

`transformers` 5.x is **required** — the legacy tuple KV-cache format is gone and all cache
manipulation here goes through `Cache` objects.

### Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

```bash
.venv/bin/python -m pytest -m "not slow"
```

101 fast tests. `-m slow` additionally runs live-model RoPE checks (needs the model).

The target model (~7.5 GB) downloads from Hugging Face on first use. **No weights are
distributed in this repository.**

### Regenerating results

Every table and report is generated from committed raw JSON — no hand-editing. These
commands re-derive the published tables **without a GPU or the model**:

```bash
.venv/bin/python scripts/make_tables.py --results results/raw/stage1_baseline_drawA.json
```

```bash
.venv/bin/python scripts/make_tables_s2a.py && .venv/bin/python scripts/make_tables_s2b.py
```

Stage 3's report is generated in **two steps, in this order** — the second appends the
detailed sections:

```bash
.venv/bin/python scripts/stage3_report.py && .venv/bin/python scripts/stage3_report_details.py
```

Verify the Stage 3 artifacts against their frozen hashes (requires the local checkpoint):

```bash
.venv/bin/python scripts/check_stage3_results.py
```

### Re-running the experiments

Approximate wall-clock on an M4 Max. Stage 2b onward needs the trained sidecar, which is
**not** committed (2.26 GB). Either retrain it (~108 min) or download the exact checkpoint
these results were produced with:

**🤗 [`jkillay/context-cache-lab-stage2b-extractor`](https://huggingface.co/jkillay/context-cache-lab-stage2b-extractor)**

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('jkillay/context-cache-lab-stage2b-extractor', 'stage2b_extractor.pt',
                      local_dir='results/raw'))"
```

That repo carries the sidecar in two forms. `stage2b_extractor.safetensors` is the one to
load — it is safe and verified bit-exact. `stage2b_extractor.pt` is a **pickle** and is
published only because its SHA-256 is what `results/stage3/freeze.json` pins, so
`check_stage3_results.py` reproduces the integrity audit exactly; the repo's scripts load
it with `weights_only=False`, so take it only if you need that verification.

| Step | Command | Time |
|---|---|---|
| Model probe | `scripts/probe_model.py --load-weights` | ~1 min |
| Build corpora | `scripts/build_corpus.py --draw A` (then `B`) | ~2 min |
| Stage 1 | `scripts/run_native_baseline.py --draw A` | ~45 min |
| Stage 2a | `scripts/run_stage2a.py --lengths 4096,16384 --limit-docs 4` | ~2 h 40 min |
| Stage 2b smoke | `scripts/smoke_stage2b.py --steps 200 --docs 2` | ~12 min |
| Stage 2b train | `scripts/train_stage2b.py --steps 2200 --lr 5e-5` | ~108 min |
| Stage 2b eval | `scripts/run_stage2b_eval.py --limit-docs 8` | ~35 min |
| Stage 2b cost | `scripts/measure_stage2b_cost.py` | ~10 min |
| Stage 3 | see [`docs/STAGE3_PROTOCOL.md`](docs/STAGE3_PROTOCOL.md) | ~59 min |

Held-out draw B is protected in code: `run_stage2b_eval.py` refuses `--draw B` without an
explicit acknowledgement flag, and the Stage 3 evaluator refuses to run if the corpus or
checkpoint hashes differ from `freeze.json`.

---

## Repository structure

```
context-cache-lab/
├── README.md              this file
├── AGENTS.md              full working record: status, failures, results, next steps
├── LICENSE                Apache-2.0
├── NOTICE                 third-party attribution and data provenance
├── CITATION.cff
├── pyproject.toml
│
├── docs/
│   ├── EXPERIMENT.md        protocol + FROZEN success criteria (§7 Stage 2b, §7b gates)
│   ├── STAGE2B_DESIGN.md    sidecar design + the eight training-objective questions
│   ├── STAGE3_PROTOCOL.md   frozen Stage 3 composition protocol (hash-verified)
│   ├── RELATED_WORK.md      literature survey; published (§5) vs. measured (§6) vs. gaps (§7)
│   └── FUTURE_IDEAS.md      deferred ideas, each with an explicit precondition
│
├── src/ccl/
│   ├── target.py            frozen target wrapper; explicit position/cache control
│   ├── rope.py              pre-RoPE capture, page composition, position reassignment
│   ├── compressor.py        C²KV-style MemoryExtractor + ratio-1.0 IdentityCompiler
│   ├── train.py             compression-concatenation co-training
│   ├── compress.py          training-free baselines and controls (Stage 2a)
│   ├── corpus.py            deterministic synthetic corpus, 9 fact classes
│   ├── stage3_corpus.py     cross-page compositional corpus
│   ├── stage3_eval.py       frozen Stage 3 evaluator
│   ├── metrics.py           distractor-aware scoring, rank margins
│   └── timing.py            MPS-synchronized timing
│
├── scripts/                 entry points (probe, build, run, train, eval, report)
├── tests/                   101 fast tests + live RoPE checks
└── results/
    ├── raw/                 raw JSON per stage + generated corpora
    ├── tables/              generated CSV/Markdown for Stages 1–2b
    └── stage3/              Stage 3 freeze, per-row JSONL, gate, report
```

---

## Preregistration and integrity

- **Criteria frozen before evaluation.** [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) §7 and
  §7b, and [`docs/STAGE3_PROTOCOL.md`](docs/STAGE3_PROTOCOL.md), were written before the
  relevant runs. Amendments are appended with dates; originals stay visible.
- **Gates are computed, not argued.** The table generators exit non-zero on failure, so a
  verdict cannot be talked past in prose.
- **Held-out means held out.** Draw B was untouched until its single final evaluation.
  Training corpora are asserted to share zero fact values with draw B and zero chunks with
  the draw-A evaluation set.
- **Hash-pinned provenance.** [`results/stage3/freeze.json`](results/stage3/freeze.json)
  records SHA-256 for the checkpoint, corpora, evaluation sources, protocol, and all
  historical Stage 1–2b result files. `check_stage3_results.py` re-verifies them.
- **Controls on every quality number.** NATIVE, NOCTX, RANDOM (moment-matched noise),
  WRONGPAGE, SHUFFLE, JOINT, and BUDGET at matched budget.
- **Every result row** records model revision, checkpoint hash, seed, draw, git commit and
  timestamp.

Known limitations are recorded alongside results, not omitted — including small per-class
samples (n=8) in Stage 2b, Stage 3's 13/24 primary validity, and the fact that Stage 2b's
corpus places nearly every fact inside a single chunk, which is why `SHUFFLE_PAGES` could
not test order sensitivity there.

---

## License and attribution

Code and documentation: **Apache-2.0** (see [`LICENSE`](LICENSE)). Chosen for compatibility
— `src/ccl/rope.py` imports `rotate_half` from Hugging Face Transformers (Apache-2.0) and
mirrors its rotary-embedding arithmetic so re-applied positions are bit-compatible.

- **No model weights are distributed in this git repository.** `Qwen/Qwen3-4B` is
  Apache-2.0 and is downloaded by the user at run time. The trained *sidecar* is published
  separately on the [Hugging Face Hub](https://huggingface.co/jkillay/context-cache-lab-stage2b-extractor)
  under Apache-2.0; it contains no Qwen weights.
- **No C²KV source code was copied.** The compressor is a reimplementation from the
  published mechanism.
- **All corpora are synthetic**, generated deterministically from integer seeds — no
  scraped text, no personal data, no third-party copyrighted material.

Full detail in [`NOTICE`](NOTICE).

## Citation

See [`CITATION.cff`](CITATION.cff). If you cite this study, please also cite
[C²KV (Du et al., KDD 2026)](https://arxiv.org/abs/2607.17715), whose mechanism it
reimplements and measures.

## Relationship to `epitaxy`

A separate cross-model KV-transfer experiment by the same author. This repository does not
import its code or mix its measurements. Stage 2b independently reproduced one qualitative
observation from it — that approximate context preserves semantics better than arbitrary
exact strings — which was preregistered here as a hypothesis to test, not an expectation.
Stage 3 did **not** reproduce that advantage on its own corpus, where every compiled class
scored zero.
