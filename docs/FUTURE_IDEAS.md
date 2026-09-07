# Future ideas — explicitly deferred

**Rule for this file: nothing here gets implemented until the Stage 1–3 baseline works.**

Every item below is a plausible improvement to a system that does not yet exist. Adding
any of them early would confound the one measurement this project is actually trying to
make — whether independently compiled context state composes at all. Each entry records
*what it is*, *what it might buy*, and *the specific precondition that must be met first*.

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

**Precondition:** measured evidence that outlier channels in `Z` are what limits
quantization — i.e. a quantization sweep that shows a specific failure mode this fixes.

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
