# Future ideas — explicitly deferred

**Rule for this file: nothing here gets implemented until the baseline it depends on
works.** Each entry states its own precondition; that precondition is the gate, not a
suggestion. The historical Stage 2c suggestion below is superseded by the 2026-09-07
Stage 3 decision: test genuine composition first with the frozen carrier.

Every item below is a plausible improvement to a system that does not yet exist. Adding
any of them early would confound the one measurement this project is actually trying to
make — whether independently compiled context state composes at all. Each entry records
*what it is*, *what it might buy*, and *the specific precondition that must be met first*.

**2026-09-08 outcome:** Stage 3 **FAIL**. Independent and joint quality are both at
floor even with two relevant pages; the interference ratio is undefined. Axis B
remains blocked. A retrieval/landmark remedy is not established, and no semantic-over-
exact generated-answer advantage appears on these new tasks. Preserve the negative
result. Any further diagnosis must be separately requested and preregistered on a new
draw; do not implement any deferred optimization below to rescue this gate.

---

## Axis B: bits per state — deferred until a clean Stage 3 pass

**Deferred.** Stage 2b is INCONCLUSIVE. Stage 3 tests cross-page composition without
quantization or further training. A clean Stage 3 pass may justify Axis B next
(chronological Stage 4 naming preferred). Interference instead motivates a separate
retrieval/landmark-selection experiment; exact-terminal failures may later motivate
a raw-page tier. None is implemented during Stage 3.

Keep the two compression axes strictly separate (see `RELATED_WORK.md` §3):

- **Axis A — state count.** `N` context positions → `M` learned memory positions. Reduces
  *attention work and active positions*. **That is Stage 2b.**
- **Axis B — bits per state.** BF16 `Z` → INT8/INT4/INT2/VQ. Reduces *bytes and
  bandwidth*, same number of positions.

If Stage 2b works in full precision, ask whether each learned memory state can also be
compressed in the representation dimension. Compare `Z_BF16` against:

1. plain INT8, then INT4 (the honest baseline — try this before anything clever);
2. **QJL**-style JL-projection + sign quantization;
3. **PolarQuant**-style random preconditioning + polar transform, quantizing angles;
4. **CommVQ**-style additive vector quantization with a learned codebook.

**The RoPE constraint that matters here.** Our `Z` is stored **pre-RoPE** and has position
re-applied at composition. A codec that does not commute with rotation would have to be
applied *after* positioning, which destroys the reusability of the stored artifact.
CommVQ's construction gives the fix: 2×2 blocks of the form `[[x, y], [−y, x]]` commute
with RoPE's rotation blocks. **Constrain any Stage 2c codec to that form** and the two
axes stay independent — the stored code can be positioned at composition time without
being dequantized first.

**Arithmetic discipline.** If Axis A reaches 8× and Axis B reaches 4×, the *storage and
traffic* reduction may approach 32×. **Do not report that product as an achieved
speedup.** It is a hypothesis until both components are measured, and it says nothing
about latency: Axis A removes attention work, Axis B removes bytes, and on unified memory
those are not the same bottleneck.

**Precondition:** a clean Stage 3 composability pass using the frozen full-precision
carrier, in addition to its historical advantage over `BUDGET`. Quantizing a
compressor that does not work would confound "the learned state is lossy" with "the
quantizer is lossy".

---

## Hadamard / orthogonal rotations

Apply a fixed orthogonal (e.g. Hadamard) transform to the latent state before storage.

**Possible uses:** spreading latent energy across dimensions; suppressing outlier
channels; enabling low-bit quantization of `Z`; compact retrieval fingerprints;
structured latent coding.

**Explicit caution:** a Hadamard transform is a rotation. It is information-preserving
and **provides no semantics**. It can make a representation *more quantizable*; it
cannot make it *more meaningful*. Any claim that rotation "improves" a representation
must be attributed to the downstream quantizer, not the rotation.

**The literature now says exactly this, which both supports the idea and bounds it.**
QJL shows that a random projection spreads outlier energy so effectively that 1-bit sign
quantization becomes viable *and* per-block scale/zero-point constants can be dropped
entirely. PolarQuant makes the same argument through a polar transform, and explicitly
connects it to random Hadamard preconditioning. So: preconditioning earns its keep
**only through a quantizer**. It is a Stage 2c component, not a standalone idea.

**Precondition:** measured evidence that outlier channels in `Z` are what limits
quantization — i.e. a quantization sweep that shows a specific failure mode this fixes.

---

## Hierarchical multi-fidelity pages

A page eventually carries more than one fidelity tier:

```
P_i = ( h_i , r_i , Z_i^low , Z_i^residual , C_i )
        │     │     │         │              └── exact raw tokens (lossless)
        │     │     │         └───────────────── optional higher-fidelity residual
        │     │     └─────────────────────────── highly compressed semantic state
        │     └───────────────────────────────── cheap retrieval landmark
        └─────────────────────────────────────── stable content ID
```

The query loads `Z^low` for everything it might need, pulls `Z^residual` only for pages
that look load-bearing, and falls back to `C_i` only when exactness is actually required.

**Motivated by, and partly pre-empted by, published work.** QuantSpec already stores one
hierarchical cache where an INT4 tensor serves a low-precision tier and residual bits
reconstruct an INT8 view, avoiding a second copy — the residual-tier layout is theirs.
LycheeMemory's JIT recompression and CacheBlend/EPIC's partial recomputation are the same
instinct applied to different tiers. What would be ours, if anything, is combining a
residual fidelity ladder with **content-addressed identity** and a **lossless raw-token
bottom tier**, then measuring what fraction of pages ever need each tier.

**Precondition:** Stage 2b must produce a working `Z` first — there is no `Z^low` to
refine without one — and Stage 7 must show that oracle exact-fallback has real headroom.
Two tiers only become interesting once one tier is proven insufficient *and* the failures
are concentrated rather than uniform.

---

## Residual vector quantization (RVQ)

Discrete, compositional coding of the compiled state:

    z ≈ E_1[c_1] + E_2[c_2] + ... + E_S[c_S]

**Possible uses:** a compact machine-to-machine codebook; large storage reduction beyond
what dense low precision gives; discrete page IDs that double as retrieval keys;
composable arithmetic over codes.

**Precondition:** a working continuous `Z` with a known quality/ratio curve. RVQ is a
*compression* of `Z`; without the curve there is no baseline to lose against. Also note
`KV Cache Transform Coding` (ICLR 2026) already treats KV as a codec problem — re-read it
before claiming anything here.

---

## Hyperdimensional / vector-symbolic codes

Binding/bundling operations over high-dimensional vectors as a structured representation
for landmarks or compositional memory.

**Possible uses:** cheap composition of page identities; superposition of many pages in
one vector; binding fact-role pairs.

**Precondition:** Stage 6 must first show that *simple* retrieval signals (embeddings,
BM25) are insufficient. Do not reach for VSA to solve a problem BM25 already solves.

---

## Engram-like rarity tables

A cheap corpus statistic over n-gram rarity, used as an exactness-risk or retrieval
signal — flagging spans containing rare tokens likely to be identifiers.

**Possible uses:** the non-oracle routing policy in Stage 7; a cheap prior on which
pages will need exact fallback.

**Explicit caution:** **do not mislabel a simple n-gram rarity table as the Engram
architecture.** If this repo builds a rarity table, the docs call it a rarity table.

**Precondition:** Stage 7 oracle fallback must show meaningful headroom over the
compressed-only condition. If the oracle barely helps, there is nothing to route.

---

## Cross-model KV transfer

Whether a page's compiled state can be produced by a *small* model and translated into
*target*-compatible state.

**Status:** tested independently in the separate `epitaxy` repo. **Do not import its
code and do not mix its measurements into this repo's results.** The two experiments
must stay scientifically separable; at most, borrow an evaluation *idea* conceptually
and cite it as such.

**Precondition:** this repo's same-model compiler must work first. A cross-model
translator that fails is uninterpretable if the same-model case was never established.

---

## KV Prediction

Compare a learned predictor of target KV ([Apple, arXiv:2410.08391](https://arxiv.org/abs/2410.08391))
against the page compressor. Both attack TTFT; they make different bets — prediction
approximates the *full* cache from a cheaper model, compression produces a *smaller*
cache from the same model.

**Precondition:** Stage 4 amortization curve exists, so the two can be compared on
matched cost axes rather than on headline speedups.

---

## Recurrent depth

Separate architecture question (latent iterative computation). Not related to the
context-compilation hypothesis. Listed only so it does not get smuggled in as "an
improvement to the compressor."

---

## Mixture of Experts

Separate compute-routing question. Same note as above.

---

## Diffusion

**Do not add unless there is a specific denoising objective justified by measured state
error.** If Stage 2/3 produce a characterized error distribution on `Z` — and that
distribution actually looks like additive noise on a manifold — a denoiser becomes a
defensible idea. Absent that measurement it is decoration.

---

## The learned context codec (the one optional experiment)

Only after the simpler compressed-KV baseline exists. Ask whether the page state needs
to resemble ordinary target KV at all:

    C_i → Z_i ∈ R^{m × d_z}          (compressor, free to choose its own latent space)
    Z_i W_A → target-readable latent memory   (adapter into the frozen target)

The compressor and adapter could in principle learn a machine-to-machine language with
no relationship to text tokens. This is the most interesting idea in the project and the
easiest one to fool yourself with.

**Preconditions, all required:**
1. Stage 3 composability established.
2. A known quality/ratio curve for the KV-shaped carrier, to beat.
3. Re-read `KV Cache Transform Coding` (ICLR 2026) and the 500xCompressor line — the
   codec framing is partly pre-empted.
4. Controls from `EXPERIMENT.md` §4 applied unchanged. A free-form latent space makes
   answer leakage *easier*, not harder, so `WRONGPAGE` and `RANDOM` matter more here
   than anywhere else.
