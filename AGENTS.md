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

**Stage 2 — not started.** This is the next piece of work.

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

**Reporting rule:** no quality number without its controls (`docs/EXPERIMENT.md` §4), and
no speedup without cold cost, warm cost, and the amortization curve.

## 8. Next steps

1. **Stage 2 — single-chunk compressor.** Build a C²KV-style sidecar: shared learnable
   compression-token embedding + per-layer QKV projection heads over the frozen target's
   hidden states. Enforce the three attention constraints from `docs/RELATED_WORK.md`
   §1.1 (original-token invariance, block-local extraction with a sink block, causal
   accumulation across compression tokens). Store KV **pre-RoPE**.
2. Add the `BUDGET` control (raw context truncated to the same token count as `Z`)
   *before* reporting any Stage 2 quality number — it is the comparison that decides
   whether compression is doing useful work.
3. Sweep 2×/4×/8×/16×. Expect degradation past ~8×; Cartridges reports generic
   compression falling apart past ~2×, so collapse is a publishable outcome, not a bug.
4. Only after Stage 2 clears its gate: Stage 3 independent compilation + composition,
   with `JOINT`, `RANDOM`, `SHUFFLE`, `WRONGPAGE` controls.

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
