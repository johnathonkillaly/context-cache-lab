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

Always use `.venv/bin/python`. The system Python is 3.14 and has **no torch**.

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

**Stage 1 — in progress.** Model probe, deterministic synthetic corpus, NATIVE/NOCTX
baselines at 1K–16K.

Stages 2–8: not started. Do not start Stage 2 until the Stage 1 gate passes
(`NATIVE` clearly beats `NOCTX` on every fact class, and `NOCTX` is near floor on the
exact classes).

## 6. Important failures and gotchas

*Recorded as they happen. An empty section here after real work would itself be a smell.*

- **Env:** system Python is 3.14.7 with no torch and no MLX. torch has no 3.14 wheels in
  this setup — the venv is pinned to 3.12.
- **transformers 5.x:** the legacy `past_key_values` tuple format is removed. KV cache
  manipulation must go through `Cache` / `DynamicCache` objects. Any snippet copied from
  a v4-era paper repo (including C²KV's) will need porting.
- **Cached `Qwen3-4B-Instruct-2507` is incomplete** (shard 3 of 3 only). Loading it will
  fail confusingly.
- **MPS timing requires explicit synchronization** (`torch.mps.synchronize()`) before
  reading the clock, or every latency number is wrong (async dispatch).

## 7. Current results

*None yet — Stage 1 baselines not yet run.* Raw output lands in `results/raw/`,
generated tables in `results/tables/`.

**Reporting rule:** no quality number without its controls (`docs/EXPERIMENT.md` §4), and
no speedup without cold cost, warm cost, and the amortization curve.

## 8. Next steps

1. Run `scripts/probe_model.py` and commit `results/raw/model_probe.json`.
2. Build corpus draws A and B; assert value pools are disjoint.
3. Run NATIVE + NOCTX at 1K/2K/4K/8K/16K; record per-fact-class accuracy and
   `rank_margin`.
4. **Evaluate the Stage 1 gate.** If `NOCTX` is high on exact classes, regenerate the
   corpus before doing anything else.
5. Only then: Stage 2 single-chunk compressor at 2×/4×/8×/16×.

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
