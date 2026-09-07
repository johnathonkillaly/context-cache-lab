# Stage 2b — design and training-objective analysis

Written **before** the compressor was implemented and **before** any training run.
Gate thresholds live in [`EXPERIMENT.md`](EXPERIMENT.md) §7b and are frozen.

The question, and only this question:

> **Can a learned full-precision compressed context state beat equal-budget raw context?**

Axis A only — state *count*, `N` context positions → `M` memory positions. Precision
(Axis B) is Stage 2c and must not be mixed in; see [`RELATED_WORK.md`](RELATED_WORK.md) §3
and [`FUTURE_IDEAS.md`](FUTURE_IDEAS.md).

---

## 1. Architecture — the smallest faithful C²KV sidecar

Following C²KV ([`RELATED_WORK.md`](RELATED_WORK.md) §1.1) as closely as MPS allows.

```
chunk tokens  c_1 … c_N            ──►  frozen target, layer ℓ  ──►  h^ℓ_1 … h^ℓ_N
                                                                        │
memory tokens m_1 … m_M  (M = N/r)  ──►  shared learnable embedding      │
                                          │                             │
                                          ▼                             │
                              per-layer Q/K/V projection heads ◄─────────┘
                                          │
                                          ▼
                              Z = { (K^ℓ_mem , V^ℓ_mem) }  ── stored PRE-RoPE
```

**Components and their sizes** (Qwen3-4B: 36 layers, `d=2560`, 8 KV heads, head_dim 128,
so `d_kv = 8 × 128 = 1024`):

| component | shape | params |
|---|---|---|
| shared memory-token embedding | `[1, 2560]` | 2,560 |
| per-layer `W_Q` | 36 × `[2560, 4096]` | 377.5 M |
| per-layer `W_K` | 36 × `[2560, 1024]` | 94.4 M |
| per-layer `W_V` | 36 × `[2560, 1024]` | 94.4 M |
| **total** | | **≈ 566 M (14% of the 4.02 B target)** |

That is close to C²KV's reported ~10% overhead. **If this proves too slow to train on
MPS, the fallback is to share one projection set across groups of layers** — recorded
here so that a reduced variant is an explicit, reported deviation rather than a silent
one.

**Structured attention flow** — the three C²KV constraints, enforced by mask only:

1. **Original-token invariance.** Chunk tokens attend only to chunk tokens, standard
   causal mask. They *never* attend to memory tokens. This is what leaves the frozen
   target's own representations bit-identical to a normal forward pass, and is why the
   base model can stay frozen.
2. **Block-local extraction.** Memory token `m_j` attends to the chunk tokens of its own
   block `B(j)` plus a **sink block** `B(0)` (the first few tokens). Stage 2a is direct
   evidence the sink matters: `KV_SINK` beat `KV_STRIDE` at every ratio it was nonzero.
3. **Causal accumulation.** `m_j` may attend to `m_1 … m_{j-1}`.

**Positional handling.** At extraction, memory token `m_j` takes the position index of
the **last chunk token in its block**. That assignment is used *only* to run the
extraction forward pass. The stored `K` is captured **before RoPE is applied**, and RoPE
is re-applied at composition time from the position the page actually lands at. `V` is
unaffected by RoPE and is stored as-is.

## 2. The eight questions

**1. What receives gradients?**
Only the shared memory-token embedding and the per-layer Q/K/V projection heads —
≈566 M parameters, ≈14% of the target. Nothing else.

**2. What remains frozen?**
The entire target model: all 4.02 B parameters, embeddings, all attention and MLP
weights, and the LM head. Enforced in code by `requires_grad_(False)` at load
(`TargetModel.__init__`) plus an explicit assertion in the training loop that no target
parameter has a gradient after `backward()`. A test asserts the trainable-parameter set.

**3. What loss is used?**
A single supervised objective on the **answer tokens only**:

```
L = − Σ_t log p( y_t | y_<t , q , K_cat , V_cat )
```

where `K_cat, V_cat` is the composed memory state with RoPE re-applied at composition
positions. No autoencoding or reconstruction loss, matching C²KV. Cartridges reports that
a plain next-token objective over the corpus is *not competitive* with in-context
learning, so reconstruction is deliberately not used as the primary signal.

**4. Does the compressor see multiple chunks during training?**
Yes — but each chunk is **encoded independently**. A training example takes `k` chunks
from one document, runs the extractor separately on each (no cross-chunk attention at
extraction), then concatenates the resulting `Z_i` before the loss. This is C²KV's
"compression-concatenation co-training", and it is the mechanism that stops the model
learning states that are individually useful but non-composable.

**5. How is concatenation/composition represented during training?**
Exactly as at inference, and via the same code path:
`compose(Z_1 … Z_k, start_position)` → re-apply RoPE per page at its actual composed
offset → splice into a `DynamicCache` → prefill the question suffix. Training and
evaluation share this function so the two cannot drift apart.

**6. Does training explicitly optimize composed state?**
Yes. Extraction is independent; **supervision is applied only after concatenation**.
Gradients flow back through the composition to every chunk's extractor. Chunk order and
the number of chunks are randomized per example so the extractor cannot learn a fixed
layout. This is the single most important design decision in Stage 2b — it is what makes
the result relevant to Stage 3.

**7. How many trainable parameters are added?**
≈566 M (table above), ≈14% of the target. Reported exactly in the results JSON.

**8. What prevents the sidecar from learning evaluation-specific answers?**
Six mechanisms, of which the first two are the load-bearing ones:

- **The compressor never sees the question.** `Compile(C_i)` is a pure function of the
  chunk. There is no query-conditioning path in the architecture at all, so
  question-specific shortcuts are not merely discouraged, they are unrepresentable.
- **Draw B is untouched.** Training uses draw A only. Draw A and draw B have provably
  disjoint value pools (asserted in `tests/test_corpus.py`), so no gold or distractor
  string in the held-out set has ever been seen.
- Within draw A, training and validation documents are split by `doc_id` with no overlap.
- `RANDOM` and `WRONGPAGE` controls run on every evaluation. `WRONGPAGE` must collapse to
  `NOCTX`; if it does not, the question itself carries the answer.
- The corpus is synthetic with `NOCTX = 0.000` and rank margins ≈0 on exact classes, so
  there is no parametric memory to leak through.
- The sidecar cannot emit tokens. It only produces KV; all decoding is done by the frozen
  target.

## 3. Smoke test before any real training run

A long run is not started until a tiny overfit experiment shows the mechanism can learn
at all. On a handful of draw-A documents, with everything else identical:

1. `L_compressed` decreases substantially from its value at initialization.
2. `rank_margin` on the *training* documents improves from ≈0 toward the `NATIVE`
   reference (+13.8 semantic).
3. The ratio-1.0 identity property still holds after training.

If (1) and (2) fail on data the model is allowed to memorize, the architecture or the
plumbing is broken, and no amount of compute will fix it. **Overfitting here is the
desired outcome, not a warning sign** — it is the positive control for the training path.
Generalization is measured separately, on held-out draw-A documents.

## 4. Positional tests that must pass

Guarding the four failure modes that would silently invalidate Stage 3:

| test | guards |
|---|---|
| stored `K` differs from post-RoPE `K` | **pre-RoPE storage** — that we captured before rotation |
| composing at offset 0 reproduces extraction-time rotation | **correct re-application** |
| same `Z` composed at different offsets gives different `K`, related by the rotation between those offsets | **positions are genuinely re-applied**, not baked in |
| applying RoPE twice is detectably different from once | **no accidental double-RoPE** |
| ratio-1.0 identity: split/independent compilation ≈ native | end-to-end sanity, within bf16 accumulation noise (measured: ≤1.0 on logits of scale 55) |
| page concatenated at arbitrary positions | arbitrary composition offsets |

The Stage 2a `SHUFFLE_KV_ORDER` no-op is retained as an informative control **but is not
a composition test** — it is provably vacuous (attention is permutation-invariant over
keys). Composition must be tested by *re-positioning*, not by reordering tensors.
