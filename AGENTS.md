# AGENTS.md — working state for Claude Code / Codex

Written for either agent to pick up cold. Keep this file current: **update Status,
Failures, Results, and Next steps in the same commit as the work they describe.**

---

## 1. Experiment goal

Determine whether long context can be **compiled once** into compact, **independently
generated**, **composable** model-readable state `Z_i`, such that a **frozen** target LLM
consuming `[Z_1;…;Z_k] + q` answers similarly to one consuming `[C_1;…;C_k] + q` —
without native prefill over the original tokens.

The deliverable is a decision: *does this architecture deserve another experiment?*
It is not a system, and it is not a benchmark win. **A failed stage gate is a valid
result and is reported as one.**

Full protocol and **frozen** success criteria: [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md).
Do not edit the criteria in §7 of that file. If they must change, append to §10 with a
date and a reason and leave the originals visible.

## 2. Architecture choices (fixed, and why)

Derived from [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) before any code ran.

| Choice | Rationale |
|---|---|
| Target model **frozen**; train only a sidecar | C²KV; ICAE/500xCompressor |
| Carrier is **per-layer KV**, not last-layer embeddings | KV carriers retain detail far better at ratio (500xCompressor vs. ICAE) |
| Store `Z` **pre-RoPE**, re-apply position at composition | C²KV's explicit answer to composing independently-encoded chunks |
| Compression tokens grouped after chunk tokens; block structure via **attention mask only** | C²KV implementation detail; keeps kernels simple |
| Original tokens **never attend to** compression tokens | preserves base-model behaviour → base can stay frozen |
| Supervise on the **answer after concatenation**, not next-token on the corpus | Cartridges reports naive next-token training is not competitive with ICL |
| Start at 2–8× compression | Cartridges: generic compression degrades fast past ~2×; C²KV: 4–16× usable |
| PyTorch/MPS, not MLX, for the core | needs direct KV-cache internals and custom attention masks |

**Non-negotiable:** compilation is a pure function of the chunk — `Z_i = Compile(C_i)`.
The compressor never sees the question. Violating this makes Stages 3–7 meaningless.

## 3. Commands

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

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

Always use `.venv/bin/python`. The system Python is 3.14 and has **no torch**.

Useful flags while iterating: `run_native_baseline.py --lengths 1024 --limit-docs 4
--skip-timing` runs in about a minute. `probe_model.py --load-weights` additionally
verifies the live RoPE module and the concrete `Cache` class a real forward pass
produces — the two things `config.json` cannot tell you and that Stage 2 depends on.

`make_tables.py` exits non-zero if the Stage 1 gate fails.

## 4. Environment and model versions

| | |
|---|---|
| Machine | Apple M4 Max, 128 GB unified memory, Darwin 27.0.0 |
| Python | 3.12.14 (`.venv`, created by `uv`) |
| torch | 2.14.0, MPS available |
| transformers | 5.16.1 — **v5 API: legacy tuple KV cache is gone, use `Cache` objects** |
| Target model | `Qwen/Qwen3-4B` @ `cdbee75…`-era snapshot, cached locally (7.5 GB, 3 shards) |
| Debug model | `Qwen/Qwen3-1.7B`, cached locally (3.8 GB) |
| Disk free | ~185 GB |

**Do not download large checkpoints without checking `~/.cache/huggingface/hub` first.**
Note: the cached `Qwen3-4B-Instruct-2507` snapshot is **incomplete** (110 MB, shard 3 of
3 only) — use `Qwen/Qwen3-4B` unless you intend to finish that download.

### Qwen3-4B geometry (from `config.json`; re-verified by `scripts/probe_model.py`)

| Field | Value |
|---|---|
| hidden_size | 2560 |
| layers | 36 |
| attention heads | 32 |
| **KV heads** | **8** (GQA, 4:1) |
| head_dim | 128 |
| RoPE theta | 1,000,000 |
| rope_scaling | none |
| max_position_embeddings | 40,960 |
| sliding window | none |
| vocab | 151,936 |
| tie_word_embeddings | true |
| dtype | bfloat16 |

KV per token = `2 × 36 layers × 8 kv_heads × 128 head_dim × 2 bytes` = **147,456 B/token
(144 KiB)**. At 16K tokens that is **2.25 GiB** of KV cache. This is the quantity the
project is trying to shrink.

## 5. Current status

**Stage 0 — complete.** Repo, env, docs, frozen pre-registration.

**Stage 1 — complete, gate PASSED.** Model geometry verified against a live forward
pass; deterministic corpus built for draws A and B (40 docs / 360 questions each,
A/B value pools asserted disjoint); NATIVE and NOCTX measured on draw A at 1K–16K
(720 evaluations).

**Stage 2a — complete.** Training-free compression floor measured (8 documents at 4K and
16K, 1,368 evaluations). Result below; it changes what Stage 2 has to beat.

**Stage 2b (learned compressor) — not started.** This is the next piece of work.

**Draw B has deliberately not been run.** It is held out for the single final
evaluation against the frozen criteria. Do not run any condition against draw B — not
even a baseline — until the compiled conditions exist.

## 6. Important failures and gotchas

*Recorded as they happen. An empty section here after real work would itself be a smell.*

- **Env:** system Python is 3.14.7 with no torch and no MLX. torch has no 3.14 wheels in
  this setup — the venv is pinned to 3.12.
- **transformers 5.x:** the legacy `past_key_values` tuple format is removed. KV cache
  manipulation must go through `Cache` / `DynamicCache` objects. Any snippet copied from
  a v4-era paper repo (including C²KV's) will need porting. `Cache.crop()` with a
  *positive* argument is deprecated — use `ccl.target.crop_to()`.
- **`rope_theta` moved into the `rope_scaling` dict.** Reading `config.rope_theta`
  directly returns `0.0` for Qwen3-4B. Stage 3 re-applies RoPE at composition time, so
  silently using `0.0` (or a v4-era default of `10_000` instead of the real `1_000_000`)
  would corrupt every composed position while still running without error.
  `tests/test_target.py::test_rope_theta_is_resolved_not_defaulted` pins this.
- **Cached `Qwen3-4B-Instruct-2507` is incomplete** (shard 3 of 3 only, 110 MB). Loading
  it will fail confusingly. Use `Qwen/Qwen3-4B`.
- **MPS timing requires explicit synchronization** (`torch.mps.synchronize()`) before
  reading the clock, or every latency number measures enqueue time, not compute.
- **MPS needs a global warm-up before a timing sweep.** With only per-measurement
  warmups, the first length measured absorbs shape-specific kernel compilation: 1K
  prefill samples fell monotonically 2.25s → 1.50s across five trials, making 1K look
  *slower* than 2K. `timing_sweep()` now warms up at the largest shape first. Residual
  run-to-run variance at 16K is still roughly ±15%, so quote medians and keep the
  per-trial samples (they are stored in the raw JSON).
- **Corpus schema drift:** draws A and B must be regenerated *together*. Loading a stale
  draw now raises an explicit message instead of a bare `TypeError`.
- **Composition is the weakest fact class and it is a target-model limit, not a harness
  bug.** Qwen3-4B with thinking disabled completes hop 1 of a 2-hop question and answers
  with the bridging *person* rather than the facility. A uniform answer-type hint lifted
  a 4-doc probe from 1/4 to 3/4 (pooled NATIVE is 0.550); thinking mode reaches 4/4 but
  spends 200–400 generated tokens per answer, which is too costly and too variable.
  **For the composition class, treat `rank_margin` as the primary metric** — it is
  +4.41 for NATIVE against −1.70 for NOCTX, so the information is demonstrably present
  and used even when it is not emitted.

## 7. Current results

Raw: `results/raw/stage1_baseline_drawA.json`. Tables: `results/tables/`.
Report: `results/tables/stage1_report_drawA.md`.

### Stage 1 gate: **PASS**

`Qwen/Qwen3-4B`, draw A, 40 documents, 720 evaluations. Metric is `clean_hit` —
correct **and** not fooled by the matched distractor.

| fact class | NATIVE | NOCTX | NATIVE margin | NOCTX margin |
|---|---|---|---|---|
| semantic | 0.975 | 0.000 | +13.84 | −0.86 |
| identifier | 1.000 | 0.000 | +5.14 | −0.10 |
| number | 1.000 | 0.000 | +5.04 | +0.01 |
| date | 1.000 | 0.000 | +2.79 | +0.13 |
| hash | 1.000 | 0.000 | +2.72 | +0.07 |
| path | 1.000 | 0.000 | +2.58 | +0.00 |
| url | 1.000 | 0.000 | +2.58 | +0.05 |
| proper_noun | 1.000 | 0.000 | +8.18 | +0.06 |
| composition | 0.550 | 0.000 | +4.41 | −1.70 |
| **overall** | **0.947** | **0.000** | | |

NATIVE holds 0.917–0.958 across 1K/2K/4K/8K/16K — no long-context degradation in this
range, so quality changes at Stage 2+ are attributable to compression, not to length.

**The corpus is clean.** NOCTX is 0.000 everywhere and its rank margins sit at ≈0.00–0.13
on every exact class, meaning the model has no parametric access to these values and is
not even weakly biased toward them. That is what makes the compiled-vs-no-context
comparison meaningful.

### Native prefill cost (cold, no prefix reuse) — the number to beat

Median of 7 trials after 3 warmups plus a global warm-up, MPS synchronized.

| context | prompt tokens | prefill (s) | tok/s | TTFT (s) | KV cache |
|---|---|---|---|---|---|
| 1024 | 1177 | 1.101 | 1069 | 1.109 | 166 MiB |
| 2048 | 2201 | 2.000 | 1100 | 2.528 | 310 MiB |
| 4096 | 4325 | 4.615 | 937 | 4.555 | 608 MiB |
| 8192 | 8542 | 9.820 | 870 | 9.771 | 1201 MiB |
| 16384 | 16979 | 22.719 | 747 | 24.745 | 2388 MiB |

**22.7 s of prefill and 24.7 s TTFT for a single 16K query on a 4B model** is the cost
that motivates this whole project. Prefill throughput decays monotonically with length
(1100 → 747 tok/s) as attention cost grows — so the win from removing active positions
is *superlinear* in the positions removed, which is the effect Stage 4 has to capture.

### Stage 2a: training-free compression floor

Raw: `results/raw/stage2a_trainfree_drawA.json`. Report:
`results/tables/stage2a_report_drawA.md`. 8 documents (4K and 16K), 1,368 evaluations.

Full-budget references — these validate the harness before anything is read into the
compression numbers:

| condition | clean_hit | margin | note |
|---|---|---|---|
| NATIVE_CACHED | 0.944 | +5.36 | matches Stage 1 NATIVE (0.947) — cache reconstruction is faithful |
| RANDOM | 0.000 | +0.12 | moment-matched noise carries nothing, as it must |
| SHUFFLE_KV_ORDER | 0.944 | +5.35 | **exact no-op, as predicted** |
| SHUFFLE_TEXT | 0.889 | +4.53 | genuinely reordering chunks costs only ~5 points |

Compression sweep (`clean_hit`):

| condition | 2× | 4× | 8× | 16× |
|---|---|---|---|---|
| KV_STRIDE | 0.028 | 0.000 | 0.000 | 0.000 |
| KV_SINK | 0.056 | 0.014 | 0.000 | 0.000 |
| KV_STRIDE_COMPACT | 0.000 | 0.000 | 0.000 | 0.000 |
| **BUDGET** (raw text) | **0.556** | **0.250** | **0.139** | **0.056** |

**Four things follow, and they shape Stage 2b:**

1. **Training-free KV dropping is catastrophic — even at 2×.** The gentlest possible
   reduction takes the model from 0.944 to 0.028. Rank margins collapse from +5.36 to
   ≈+0.1, i.e. essentially no discrimination between gold and distractor. A learned
   compressor is *necessary*, not an optimization. This is consistent with C²KV
   (naive compression + non-prefix reuse degrades severely) and Cartridges (generic
   compression falls apart past ~2×).
2. **BUDGET ≈ 1/ratio, almost exactly** (0.556 / 0.250 / 0.139 / 0.056 against 0.50 /
   0.25 / 0.125 / 0.0625). That is the signature of an all-or-nothing baseline: a fact
   is either inside a retained chunk or it is gone. It makes the bar for Stage 2b
   unusually crisp — **a learned compressor is only interesting if it beats 1/r, which
   means proving that a degraded version of *every* chunk beats a perfect version of
   *some* chunks.** That is the real scientific question, and it is sharper than
   "beat no-context".
3. **`SHUFFLE_KV_ORDER` is an exact no-op**, to three decimals including margins.
   Attention is permutation-invariant over keys and a post-RoPE key carries its position
   in its own rotation, so reordering cache tensors changes nothing observable. This is
   not a null result to shrug at — it is a direct demonstration of why C²KV must store
   KV **pre-RoPE**: an independently encoded chunk cannot be re-placed without being
   re-rotated. **Do not use a KV-order shuffle as a Stage 3 control; it is vacuous by
   construction.** Use `SHUFFLE_TEXT`-style reordering, and note it only costs ~5 points
   natively, so surviving a shuffle proves less than it looks.
4. **Re-dating positions hurts.** `KV_STRIDE_COMPACT` (query re-dated to the shrunken
   length) is at or below `KV_STRIDE` (original absolute positions) at every ratio, with
   negative margins at 2× and 4× where `KV_STRIDE` is positive. Both are near floor so
   the effect is small, but the sign is consistent and it is the same failure mode
   Stage 3 is exposed to.

**Reporting rule:** no quality number without its controls (`docs/EXPERIMENT.md` §4), and
no speedup without cold cost, warm cost, and the amortization curve.

## 8. Next steps

1. **Stage 2b — learned single-chunk compressor.** Build a C²KV-style sidecar: a shared
   learnable compression-token embedding plus per-layer QKV projection heads over the
   frozen target's hidden states. Enforce the three attention constraints from
   `docs/RELATED_WORK.md` §1.1 (original-token invariance, block-local extraction with a
   sink block, causal accumulation across compression tokens). Store KV **pre-RoPE** —
   Stage 2a's `SHUFFLE_KV_ORDER` no-op is the direct demonstration of why.
2. **Target the bar Stage 2a set, not `NOCTX`.** `BUDGET ≈ 1/r`, so the compressor is
   only interesting if it beats 0.250 at 4× and 0.139 at 8×. Concretely: prove that a
   degraded version of *every* chunk beats a perfect version of *some* chunks. If it
   cannot, the honest conclusion is that chunk selection (Stage 6) matters more than
   chunk compression, and the project should pivot there.
3. Sweep 2×/4×/8×/16×. Given that training-free dropping is already at floor by 2×,
   treat any learned result above `BUDGET` at 4× as a real signal, and expect collapse
   somewhere between 8× and 16×.
4. Training data must come from draw A only, and the compressor must never see a
   question at compile time (`docs/EXPERIMENT.md` §8). Cartridges reports that a naive
   next-token objective on the corpus is not competitive with in-context learning —
   supervise on the answer *after* concatenation, as C²KV does.
5. Only after Stage 2b clears its gate: Stage 3 independent compilation + composition,
   with `JOINT`, `RANDOM`, `WRONGPAGE` and a **text-level** shuffle control.

Two efficiency notes for whoever runs the next sweep:

- `run_stage2a.py` rebuilds the context cache once per *question* rather than once per
  *condition*, so a 16K document pays 162 cache builds instead of 18. The full 8-document
  sweep took ~2h40m. Building once per condition and cropping back after each question
  (as `run_native_baseline.py` already does) would cut this substantially.
- Piping a long background run through `tail` buffers all output until exit. Use `tee`
  to a log file if you want to watch progress.

Do not touch draw B until the final held-out evaluation.

## 9. Repo conventions

- Every result row records: model revision, seed, corpus draw, git commit, timestamp.
- Greedy decoding, `temperature=0`, fixed seeds. Timings are the median of ≥5 trials
  after ≥2 warmups, with MPS synchronized.
- Raw JSON/CSV in `results/raw/` is committed. Large binaries (`.safetensors`, `.pt`) are
  gitignored.
- Plots only where they clarify a result.
- `docs/FUTURE_IDEAS.md` items stay unimplemented until their stated precondition is met.

## 10. Separation from `epitaxy`

`epitaxy` is a **separate** cross-model KV-transfer experiment. Do not import its code,
depend on it, or mix its measurements into these results. Borrowing an evaluation *idea*
conceptually is fine and must be cited as such.
