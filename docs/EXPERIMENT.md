# EXPERIMENT — pre-registration and protocol

**Frozen on:** 2026-09-07, before any model was run and before any result was seen.
Success thresholds in §7 are **not to be edited after the held-out evaluation runs.**
If they change, the change must be recorded in §10 with a date and a reason, and the
original values must remain visible.

---

## 1. Central question

Can independently chunked context

    C_1, C_2, ..., C_n

be compiled independently into compact states

    Z_1, Z_2, ..., Z_n        with |Z_i| << |C_i|

such that a **frozen** target LLM consuming `[Z_1; ...; Z_k] + q` answers similarly to
one consuming `[C_1; ...; C_k] + q`, **without native prefill over the original tokens**?

Three sub-questions, in dependency order:

1. **Survival** (Stage 2) — does target behaviour survive compression of a *single*
   chunk at all?
2. **Composability** (Stage 3) — do *independently* compiled states compose?
3. **Economics** (Stages 4, 8) — does the target-side cost actually shrink on this
   hardware, once compilation cost is accounted for honestly?

Stage 3 is the load-bearing experiment. Stages 5–7 are only worth building if it passes.

---

## 2. Hardware and models

| | |
|---|---|
| Machine | Apple M4 Max, 128 GB unified memory, macOS (Darwin 27.0.0) |
| Backend | PyTorch 2.14 / MPS. No CUDA assumptions. No FlashAttention CUDA kernels. |
| Python | 3.12.14 (venv at `.venv`, created with `uv`) |
| transformers | 5.16.1 |

**Target model:** `Qwen/Qwen3-4B` — already fully cached locally (7.5 GB, 3 shards), so
no download is required. Geometry verified in Stage 1 (`results/raw/model_probe.json`).

Chosen because (a) it is already on disk, (b) C²KV evaluates `Qwen3-4B-Instruct-2507`,
the same family and size, making a rough sanity comparison possible, and (c) it fits
comfortably in unified memory in bf16 alongside a sidecar.

**Fast-iteration model:** `Qwen/Qwen3-1.7B` (also cached, 3.8 GB) for harness debugging
only. **No headline result may be reported from the 1.7B model.**

MLX is available and may be used later for throughput comparison, but the primary
implementation is PyTorch/MPS because the experiment needs direct access to KV cache
internals and custom attention masks.

---

## 3. Evaluation corpus

Deterministic and synthetic, generated from an explicit integer seed. Rationale: we need
(a) exact ground truth, (b) controllable fact placement, (c) matched distractors, and
(d) guaranteed absence from the model's pretraining data. Natural corpora fail (d), and
a model answering from parametric memory would silently invalidate every result.

**Fact classes** (each with a matched distractor elsewhere in the document):

| Class | Example fact | Question | Scoring |
|---|---|---|---|
| `semantic` | "Fred bought a blue bicycle because his car broke." | "What color bicycle did Fred buy?" | normalized containment |
| `identifier` | "The replacement assembly is QZ-8471-BX." | "What is the exact replacement assembly?" | exact match |
| `number` | "Invoice total was 48,317 credits." | "What was the invoice total?" | exact match (normalized) |
| `date` | "The audit closed on 2029-11-04." | "When did the audit close?" | exact match |
| `hash` | "Manifest digest `a3f9c1...` (64 hex)" | "What is the manifest digest?" | exact match |
| `path` | "Logs are written to /var/opt/mirren/spool/kx19.log" | "Where are logs written?" | exact match |
| `url` | "Mirror at https://depot.example.net/rel/8821" | "What is the mirror URL?" | exact match |
| `proper_noun` | "Dr. Halvard Nieminen signed off." | "Who signed off?" | exact match |
| `composition` | facts split across two **separated** chunks | requires joining both | exact match on the joined answer |

**Distractors are mandatory.** For every target fact, at least one same-class,
similar-surface, wrong-value fact appears in a *different* chunk. Without distractors an
exact-match score is unfalsifiable — the model can be right by format-guessing.

**Context lengths:** 1K, 2K, 4K, 8K, 16K tokens. Longer only if practical on this box.

**Corpus draws:** two independent seeds. Draw A is used for all development and any
learned-module training. Draw B is **held out** and is run exactly once, at the end,
against frozen criteria. Training documents and held-out documents must never overlap.

---

## 4. Conditions and controls

Every stage that reports a quality number must report these, or state which are
inapplicable and why:

| Code | Condition | Purpose |
|---|---|---|
| `NATIVE` | `[C_1;...;C_n] + q`, full native prefill | upper bound |
| `NOCTX` | `q` alone | lower bound / parametric-memory leak check |
| `COMPILED` | `[Z_1;...;Z_n] + q`, independently compiled | the hypothesis |
| `JOINT` | `Z` compiled from the whole document at once | isolates the cost of *independence* |
| `RANDOM` | `Z` replaced by random vectors of matching shape and moment | proves `Z` carries information |
| `SHUFFLE` | correct `Z_i`, permuted order | tests whether order/position is used at all |
| `WRONGPAGE` | `Z_i` from a *different* document | proves the answer is not leaking via the question |
| `BUDGET` | raw context truncated to the same token count as `Z` | **the control most often missing in this literature** |

`NOCTX` is not a formality. If `NOCTX` scores well on a fact class, that class is
contaminated and must be regenerated.

`BUDGET` is the honest comparison. A compiled state that beats `NOCTX` but loses to the
same number of raw tokens has demonstrated nothing of value.

---

## 5. Metrics

**Quality**
- `semantic_acc` — normalized containment of the gold answer.
- `exact_acc` — exact match after whitespace/case normalization (case preserved for
  identifiers, hashes, paths, URLs).
- `answer_logprob` — mean token logprob of the gold answer under teacher forcing.
  Reported alongside generation because it is lower-variance and does not depend on
  decoding or output format.
- `rank_margin` — `logprob(gold) - logprob(distractor)`. This is the sharpest signal for
  detecting partial degradation before it shows up in accuracy.

**Cost** (all wall-clock, MPS synchronized, median of ≥5 trials after ≥2 warmups)
- `T_compile` — sidecar cost to produce `Z` from `C`. Never omitted.
- `T_prefill_native` — target prefill over raw tokens.
- `T_prefill_compiled` — target prefill over `Z`.
- `TTFT_cold` = `T_compile + T_prefill_compiled + T_first_token`
- `TTFT_warm` = `T_prefill_compiled + T_first_token`
- `peak_memory` — unified-memory footprint where practical.
- `n_active_positions` — positions the target actually attends over. **This, not disk
  size, is the quantity the project is trying to reduce.**
- `sidecar_bytes` — serialized size of `Z`.

**Ratios reported per configuration**
- active-position compression `N_raw / N_compiled`
- warm speedup `T_native / T_compiled,warm`
- amortized cost over `m` queries: `(T_compile + Σ_{j=1..m} T_j) / m`, for
  m ∈ {1, 2, 4, 8, 16, 32}

**Determinism:** greedy decoding, fixed seeds, `temperature=0`. Every result row records
model revision, seed, corpus draw, git commit, and timestamp.

---

## 6. Stage plan and gates

Each stage has an explicit **kill gate**. A failed gate is a result, and it is reported
as one — it does not mean "try harder until it works."

| Stage | Content | Gate to proceed |
|---|---|---|
| 0 | Repo, env, docs | — |
| 1 | Model geometry probe; corpus; **NATIVE + NOCTX baselines** at 1K–16K | NATIVE clearly beats NOCTX on every fact class; NOCTX near floor on exact classes |
| 2a | **Training-free** compression floor: position dropping, `BUDGET`, `RANDOM`, `SHUFFLE` | none — this is a measurement stage that sets the bar for 2b |
| 2 | Single-chunk **learned** compression at 2×, 4×, 8×, 16× | at ≥1 ratio, COMPILED > NOCTX by a wide margin **and** ≥ BUDGET **and** ≥ the Stage 2a floor |
| 3 | **Independent compilation + composition** | COMPILED ≈ JOINT (within noise), and both ≫ RANDOM/SHUFFLE/WRONGPAGE |
| 4 | Reuse and amortization | warm TTFT beats native prefill; amortization curve crosses over at reasonable m |
| 5 | Content-addressed pages `(h_i, z_i, C_i)` | dedup + recompilation-skip verified; raw recoverable byte-exact |
| 6 | Landmark / retrieval key `r_i` | retrieval recall@k measured *independently of generation* |
| 7 | Exact fallback / page repair | oracle fallback recovers a large share of exact-match failures |
| 8 | M4 Max memory/traffic analysis | — (reporting stage) |

**Stage 1 must fully pass before any compressor is written.** A baseline harness that
cannot distinguish NATIVE from NOCTX cannot evaluate anything.

**Stage 2a was inserted after Stage 1 passed** (it was not in the original plan). The
reasoning: a learned compressor is only worth building if the frozen target cannot
already be served by a heuristic, and the *floor* set by heuristics is the honest bar
for Stage 2 — not `NOCTX`. It costs no training and reuses the Stage 1 harness. It also
forced the pre-RoPE/post-RoPE distinction into the code before Stage 3 depends on it.

### Architectural commitments (from the literature survey)

- **Freeze the target model.** Train only a sidecar. (C²KV; also ICAE/500xCompressor.)
- **Store `Z` pre-RoPE; re-apply positions at composition time.** This is C²KV's answer
  to the positional problem and it is the default. Storing post-RoPE, and letting every
  chunk believe it starts at position 0, are **ablations**, not the design.
- **KV-shaped carrier, not last-layer embeddings.** The 500xCompressor / C²KV line shows
  per-layer KV retains detail far better at high ratios.
- **Do not train with a plain next-token objective on the corpus.** Cartridges reports
  this is not competitive with in-context learning. Supervise on answers after
  concatenation, as C²KV does.
- **Start at 2–8×.** Cartridges reports generic compression degrading rapidly past ~2×;
  C²KV reports usable 4–16×. Collapse at 16× would be consistent with published results
  and is not evidence of a broken harness.

---

## 7. Pre-registered success criteria — FROZEN

Evaluated on **corpus draw B (held out)**, run once.

### Promising — all six required

1. Independently compiled chunks remain composable: `COMPILED` within **5 accuracy
   points** of `JOINT` on semantic facts.
2. Compiled memory substantially outperforms no-context: `COMPILED` − `NOCTX` ≥ **25
   points** semantic accuracy.
3. **≥4×** fewer active target context positions at acceptable quality, where acceptable
   means ≥ **60%** of `NATIVE` semantic accuracy.
4. Warm repeated-query TTFT improves over native prefill by ≥ **1.5×** at 8K context.
5. Semantic information survives significantly better than no-context — i.e. criterion 2
   holds *per fact class* for `semantic` and `composition`, not only in aggregate.
6. Results reproduce on corpus draw B: every above metric within **10 relative percent**
   of draw A.

### Strongly promising — all four required, in addition to the above

- **≥8×** active-position compression at ≥ **80%** of `NATIVE` semantic accuracy.
- Exact fallback (Stage 7) recovers ≥ **70%** of `COMPILED`'s exact-match failures while
  routing ≤ **25%** of pages to raw prefill.
- Warm target-side TTFT improves by ≥ **2×**.
- `COMPILED` ≥ `BUDGET` at matched token count. *(If a compiled state cannot beat the
  same number of raw tokens, the compression is not doing useful work regardless of any
  other number.)*

### Explicit failure conditions — report and stop

- `COMPILED` ≈ `RANDOM` → the carrier holds no information.
- `COMPILED` ≈ `SHUFFLE` at all ratios → position is unused; composition is a bag of
  chunks and long-range composition cannot work.
  *(Amended 2026-09-07 after Stage 2a: a **KV-order** shuffle is a provable no-op and
  must not be used here. Use text-level chunk reordering, which itself costs only ~5
  points natively — so this criterion is weak evidence either way and is demoted to
  informational.)*
- `COMPILED` < `BUDGET` at every ratio → the compressor is worse than truncation.
- `NOCTX` high on exact classes → corpus contaminated; regenerate before anything else.

---

## 7b. Stage 2b gate — FROZEN 2026-09-07, before the compressor was written

Stage 2b asks one question:

> **Does an approximate representation of ALL chunks beat an exact representation of 1/r
> of the context?**

That makes `BUDGET` — not `NOCTX` — the primary comparator. Thresholds are set against
the score scales already measured on draw A, so they are calibrated rather than invented:

| reference (draw A, `clean_hit`) | value |
|---|---|
| `NATIVE` | 0.944 |
| `BUDGET` @2× / @4× / @8× / @16× | 0.556 / 0.250 / 0.139 / 0.056 |
| best training-free KV @2× | 0.056 |
| `NOCTX` | 0.000 |

Define the **recovery fraction** at ratio `r`:

```
recovery(r) = ( LEARNED(r) − BUDGET(r) ) / ( NATIVE − BUDGET(r) )
```

`recovery = 0` means the compressor is worth exactly as much as keeping 1/r of the raw
text. `recovery = 1` means compression is free.

### FAILED representation hypothesis — report and stop

Any of:

- `LEARNED(r) ≤ BUDGET(r) + 0.05` at **every** ratio tested; or
- `LEARNED(4×) < 0.30` (i.e. it cannot clear `BUDGET(4×)=0.250` by a usable margin); or
- `LEARNED ≈ RANDOM` at every ratio.

**Conclusion in that case:** selection / raw-page retrieval is more promising than state
compression, and the project should pivot to Stage 6 rather than adding machinery to
rescue Stage 2b. Do not raise ratios, enlarge the sidecar, or extend training in search
of a better number after seeing a failure.

### PROMISING — all four required

1. `LEARNED(4×) ≥ 0.40`, i.e. **recovery(4×) ≥ 0.22** and a clear margin over
   `BUDGET(4×)=0.250`.
2. `LEARNED(4×) − NOCTX ≥ 0.35`.
3. Mean `rank_margin` on `semantic` ≥ **+2.0** at 4× (against `NATIVE` +13.8, `NOCTX`
   −0.86) — the representation must preserve ranking, not just occasionally guess right.
4. Positional correctness holds: the ratio-1.0 identity test passes, and composed
   multi-chunk `Z` is not worse than single-chunk `Z` by more than **10 relative
   percent** on semantic facts.

### STRONGLY PROMISING — either of

- **recovery(4×) ≥ 0.70**, i.e. `LEARNED(4×) ≥ 0.736`; or
- `LEARNED(8×) ≥ 0.35`, i.e. comfortably above `BUDGET(8×)=0.139`.

### Notes on interpretation, fixed in advance

- **Per-fact-class reporting is mandatory**, not optional: `semantic`, `proper_noun`,
  `number`, `date`, `hash`, `identifier`, `path`, `url`, `composition`. The aggregate
  number alone must not be quoted.
- A separate experiment (`epitaxy`, not this repo, code and results not imported)
  observed that approximate context preserves semantics far better than arbitrary exact
  strings. **This is recorded as a hypothesis to test, not an expectation.** If our
  exact classes survive as well as `semantic`, that is a genuine disconfirmation and is
  reported as one.
- All thresholds above are frozen. Draw B remains untouched until the single final
  evaluation.

---

## 8. Anti-leakage protocol

If a learned compressor appears to work, verify it is not leaking answers through the
evaluation construction, before reporting anything:

1. **`WRONGPAGE` must collapse to `NOCTX`.** If it does not, the question itself carries
   the answer.
2. **Held-out documents share no fact values with training documents** — enforced by
   disjoint value pools per draw, asserted in tests, not by inspection.
3. **Distractor sensitivity:** swapping gold and distractor values must swap the model's
   answer. If it does not, the model is answering from format priors.
4. **Question-only ablation per fact class**, not just in aggregate.
5. The compressor never sees questions at compile time. Compilation is a pure function of
   the chunk: `Z_i = Compile(C_i)`. Any violation makes Stages 3–7 meaningless.

## 9. Final report must answer

1. Can target context be compiled once and reused?
2. Can independently compiled chunks be composed?
3. What compression ratio works before quality collapses?
4. What information is lost first?
5. Does keeping raw pages as a lossless sidecar solve exact recall economically?
6. How much native target prefill remains necessary?
7. Cold TTFT?
8. Warm TTFT?
9. How does repeated-query amortization behave?
10. Is retrieval/landmark selection worthwhile?
11. Does the result justify building a true learned context codec?
12. What part, if any, appears unexplored by the cited literature?

Question 12 is answered against [RELATED_WORK.md](RELATED_WORK.md) §5–7, which was
written **before** any result existed, specifically so that novelty cannot be
rationalized after the fact.

## 10. Amendments

*(none — criteria frozen 2026-09-07)*

### 2026-09-07 — separate Stage 3 composition experiment

User-directed continuation selects the frozen-carrier path, not further Stage 2b
training. Stage 2b remains **INCONCLUSIVE** and all historical criteria/results above
remain visible and unchanged. A usable carrier beating BUDGET motivates testing a
previously unanswered question: cross-page composition. The earlier corpus's page
shuffle did not test order-dependent facts.

The separately frozen [Stage 3 protocol](STAGE3_PROTOCOL.md) defines new development
and held-out draws, native/missing-page validity, 4× primary gates, controls, timing,
and PASS/PARTIAL/FAIL interpretation. Original held-out Draw B is not used. Its
criteria supplement this record and do not retroactively amend §7 or §7b. No further
training, quantization, retrieval, fallback, or other deferred extension is authorized
within Stage 3. Corpus and step-2200 checkpoint hashes: `results/stage3/freeze.json`.
