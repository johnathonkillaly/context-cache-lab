# Related Work

**Status:** §1–2 and §4–7 written 2026-09-07 **before any code**. §3 (the quantization /
precision-axis literature) added later the same day, after Stage 2a and **before** the
Stage 2b compressor was written.

**Bottom line up front: the central idea of this repo is not novel.** Independently
compiling context chunks into compressed, composable, model-readable state — with a
frozen target model — is exactly what **C²KV (KDD 2026)** does, and **Cartridges
(2025)** already established the "compile a corpus once, amortize across queries"
economics. **Landmark Attention (2023)** established block-level landmark retrieval.
Anything this project contributes has to be found in the gaps listed at the bottom of
this document, not in the core hypothesis.

This project should therefore be understood primarily as a **replication and
measurement study on Apple Silicon**, with a small number of genuinely open questions
attached. See [§7 Where the gaps actually are](#7-where-the-gaps-actually-are).

---

## 1. The three papers requested, in mechanism detail

### 1.1 C²KV — Compressed and Composable KV Cache Reuse for Efficient LLM Inference

Du, Chen, Tang, Liu, Lan, Qu, Niu, Liu, Chen, Wu. KDD 2026. [arXiv:2607.17715](https://arxiv.org/abs/2607.17715) ·
[code](https://github.com/s7a9/C2KV)

This is the closest published system to the hypothesis in this repo. Mechanism:

**Independent chunk prefill.** Each document is encoded on its own, with no knowledge
of the other documents it will later be concatenated with. This is precisely the
"compile `C_i → Z_i` independently" step of Stage 3.

**The Extractor sidecar.** A lightweight module that runs alongside the frozen base
LLM — *not* a full model copy and *not* LoRA. It consists of learnable C²Token
embeddings plus per-layer QKV projection heads, adding roughly **10% additional
parameters** over the base model.

**Compression tokens (C²Tokens).** For a chunk of `n` tokens at compression ratio `k`,
`m = ⌈n/k⌉` C²Tokens are used. All C²Tokens **share a single learnable embedding
vector** and carry no lexical content; one C²Token is associated with each consecutive
block of `k` document tokens. Implementation detail worth copying: the C²Tokens are
*physically* placed after the document tokens, and the block structure is enforced
**purely through the attention mask**.

**Structured attention flow** — three constraints:
1. *Original-token invariance:* original tokens attend only to other original tokens
   under the standard causal mask. Attention from original tokens to any C²Token is
   strictly disallowed. (This is what keeps the base model's own representations
   unperturbed, and it is why the base model can stay frozen.)
2. *Block-local extraction:* each C²Token `c_j` attends only to the original tokens in
   its own block `B(j)` plus a **sink block `B(0)`**.
3. *Causal accumulation:* C²Tokens may attend to preceding C²Tokens causally.

**Positional / RoPE handling — the critical part.** At extraction time each C²Token is
assigned the position index of the **last token in its block**. That assignment is used
*only during extraction*; **the resulting key/value pairs are stored before positional
embeddings are applied**. At inference the retrieved KVs receive **fresh positional
embeddings** based on where they land in the final assembled sequence, with positional
offsets accumulated across concatenated segments. The paper describes the learned
manifold as "explicitly position-agnostic."

> This directly answers the warning in our own Stage 3 brief ("do not silently let every
> chunk believe it occurs at position zero"). The published answer is: **store
> pre-RoPE, re-apply RoPE at composition time.** We should adopt this and treat
> alternatives as ablations, not as the default.

**Frozen base model.** Yes. All base parameters stay frozen; gradients flow only to the
C²Token embeddings and their QKV projection heads.

**Training objective.** A single supervised fine-tuning objective on the answer —
standard autoregressive LM loss `L_SFT = -Σ log p(y_t | y_<t, q, K_cat, V_cat)`. The
"compression-concatenation co-training" trick is that documents are **extracted
independently but supervised only after concatenation**, which forces the extractor to
produce KV that stays valid under arbitrary composition. There is no autoencoding /
reconstruction loss.

**Compression ratios.** 4×, 8×, 16× as fixed-ratio variants (`C2KV-#x`), plus
`C2KV-Dyn` trained with dynamically sampled ratios. Reported to degrade gracefully as
ratio increases, and to be substantially more robust than prior reuse methods.

**Models and data.** Base models: Qwen3-4B-Instruct-2507, Llama-3.1-8B-Instruct,
Qwen2.5-7B-Instruct, Qwen3-14B. Training: HotpotQA, 2WikiMultiHopQA, LongMagpie
(40k samples each, 120k total, single epoch). Eval: LongBench subsets (HotpotQA,
2WikiMQA, MuSiQue, MultiNews, SAMSum, QMSum, GovReport), RULER, GSM8K.

**Latency methodology — read this carefully before quoting the 17× number.** TTFT
includes loading precomputed document KVs from host memory to GPU plus lightweight
positional reassignment. **Offline document KV extraction time is excluded from TTFT.**
Decode cost is reported separately as time-between-tokens (TBT). The headline "up to
17× inference speedup under long contexts" is therefore a *warm* number.

> Our Stage 4 requirement to report cold cost, warm cost, *and* the amortized curve is
> a deliberate methodological tightening relative to this paper, not a novelty claim.

Note: C²KV already evaluates **Qwen3-4B-Instruct-2507**, which is in the same family as
the model this repo targets. That makes a direct sanity comparison possible.

---

### 1.2 Landmark Attention — Random-Access Infinite Context Length for Transformers

Mohtashami & Jaggi (EPFL). NeurIPS 2023. [arXiv:2305.16300](https://arxiv.org/abs/2305.16300) ·
[code](https://github.com/epfml/landmark-attention/)

**Block landmarks.** The input is split into fixed-size blocks; each block gets a single
**landmark token** that acts as a representative/gating token for that block. The
landmark's key vector is trained to be the thing you attend to when you want to decide
whether the block is relevant.

**How landmark tokens interact with ordinary attention — the key design choice.** Block
selection is performed *by the model's own attention mechanism*, not by a separate
retriever. A query first attends over the landmark tokens to score blocks; only the
top-scoring blocks then have their actual tokens loaded into standard attention. Because
selection and use share one mechanism, the retrieval criterion is guaranteed compatible
with what attention actually needs — the paper's central argument against bolt-on
retrieval.

**Hierarchical memory / random access.** Because only selected blocks are materialized,
the bulk of the KV cache can live in slower memory (CPU/disk) and be paged in on demand.
This preserves *random access* — the ability to reach any token in the full context —
which recurrent-memory and summarization approaches give up. The paper explicitly frames
this as integrating with "the system's memory hierarchy," which is the same L1/L2/L3
mental model this repo is built around.

**Results.** Comparable to Transformer-XL while retrieving far fewer tokens per step;
fine-tuning LLaMA-7B extended usable context beyond 32k.

**Relevance to us:** this is the reference design for Stage 6. Important caveat — it
requires **fine-tuning the base model** to train the landmark behaviour. Our Stage 6
brief deliberately starts with *non-attention* retrieval signals (embeddings, BM25),
which is cheaper but forfeits the compatibility argument above. That trade-off should be
stated explicitly in any writeup.

---

### 1.3 LycheeMemory — Dynamic Long Context Reasoning over Compressed Memory via End-to-End RL

Chen, Li, Zhang, Hu, Zhang. ACL 2026. [arXiv:2602.08382](https://arxiv.org/abs/2602.08382) ·
[ACL Anthology](https://aclanthology.org/2026.acl-long.365/) · [org](https://github.com/LycheeMem)

Three modules:

**Compressor (chunk-wise compressed memory).** Segments input into chunks and encodes
each into compressed **KV-cache-style latent representations**. Built on a base LLM
augmented with a **LoRA** module — so unlike C²KV, the encoding path is an adapted copy
of the model rather than a separate sidecar head.

**Gate (retrieval).** Scores each candidate memory block for relevance given the user
query *and the current working memory*, then selects. Trained separately, as a
classifier — i.e. retrieval is decoupled from generation, unlike Landmark Attention.

**Reasoner (working memory).** Iteratively processes selected blocks against an evolving
working memory, enabling multi-step reasoning across chunks instead of a single pass over
everything. This is the part that makes it "dynamic" — retrieval is re-run as reasoning
proceeds.

**Training.** Compressor and Reasoner are jointly optimized **end-to-end with
reinforcement learning**; the Gate is trained separately. This is a much heavier training
regime than C²KV's single SFT objective and is **out of scope for this repo's hardware**.

**Long-context extrapolation.** Reported to extrapolate from 7K to 1.75M tokens, with ~2×
lower peak GPU memory than MemAgent (2025). Evaluated on RULER-HQA, 2WikiMultihopQA,
StreamingQA. Optional Just-in-Time (JIT) recompression beyond 1M tokens, on the argument
that recomputing beats the I/O bottleneck of swapping — a useful data point for our
Stage 8 memory-traffic analysis.

**Does exact information suffer?** The paper's evaluation is multi-hop QA and
summarization, which are **semantic** benchmarks. Its own framing ("cognitively
inspired", human-like forgetting) implies lossy recall, and the JIT-recompression escape
hatch is an implicit admission that the compressed form is not always sufficient. We
found **no per-fact-class breakdown** (exact identifiers, hashes, serials, paths) in the
reported results. **This gap is real and is one of the few places our Stage 2/7 design
adds something the cited literature does not report.**

---

## 2. Comparison table

Legend — **Frozen:** base target model unmodified. **Indep:** chunks encoded without
knowledge of each other. **Compose:** independently encoded chunks concatenated at
inference. **Exact fallback:** lossless raw tokens retained and selectively re-inserted.

| System | Year | Carrier | Frozen | Indep | Compose | Positional handling | Retrieval | Exact fallback | Training cost |
|---|---|---|---|---|---|---|---|---|---|
| **C²KV** | 2026 | compressed KV, pre-RoPE | ✅ | ✅ | ✅ learned | store pre-RoPE, re-apply at composition | ❌ | ❌ | SFT, sidecar only (~10% params) |
| **Landmark Attention** | 2023 | full KV + landmark keys | ❌ finetune | ✅ blocks | ✅ (native attn) | standard | ✅ via attention | n/a (blocks are exact) | finetune base |
| **LycheeMemory** | 2026 | compressed KV (LoRA enc.) | ❌ LoRA | ✅ | ✅ + working mem | not emphasized | ✅ learned Gate | ⚠️ JIT recompress | end-to-end RL |
| **Gist Tokens** | 2023 | activations | ❌ finetune | n/a | ❌ | standard | ❌ | ❌ | finetune base |
| **AutoCompressor** | 2023 | recursive summary vectors | ❌ finetune | ❌ recursive | ❌ | standard | ❌ | ❌ | finetune base |
| **ICAE** | 2024 | last-layer embeddings | ✅ | ✅ | ⚠️ concat groups | ignores pos. effects | ❌ | ❌ | pretrain encoder |
| **500xCompressor** | 2025 | per-layer KV | ✅ | ✅ | ⚠️ | ignores pos. effects | ❌ | ❌ | pretrain encoder |
| **Cartridges (self-study)** | 2025 | trained KV cache | ✅ | ❌ whole corpus | ❌ monolithic | standard | ❌ | ❌ | **test-time training per corpus** |
| **Cartridges at Scale** | 2026 | many modular KV caches | ✅ | ✅ per-doc | ✅ | standard | ✅ + RAG | ❌ | TTT per cartridge |
| **PromptCache** | 2024 | raw KV segments | ✅ | ✅ | ✅ schema-bound | reassigned, lossy | ❌ | n/a | none |
| **CacheBlend** | 2025 | raw KV + partial recompute | ✅ | ✅ | ✅ | recompute fixes it | ❌ | ✅ **token-level** | none |
| **EPIC** | 2025 | raw KV + sink recompute | ✅ | ✅ | ✅ | sink-token recompute | ❌ | ✅ sink only | none |
| **TurboRAG** | 2024 | precomputed chunk KV | ✅ | ✅ | ✅ independent attn | reordered position IDs | ❌ | ❌ | (light finetune) |
| **KVLink** | 2025 | raw KV + link tokens | ❌ | ✅ | ✅ learned links | link tokens bridge | ❌ | ❌ | extra pretraining |
| **KV Prediction** | 2024 | predicted KV from small model | ✅ | ❌ | ❌ | standard | ❌ | ❌ | train auxiliary model |
| **Prefix caching** (vLLM/SGLang, Anthropic, Gemini) | — | exact raw KV | ✅ | ❌ prefix only | ❌ | exact | ❌ | n/a (exact) | none |
| **This repo (proposed)** | 2026 | TBD — start from C²KV-style | ✅ | ✅ | ✅ | adopt C²KV pre-RoPE | Stage 6 | ✅ **page-level, risk-routed** | sidecar only |

---

## 3. Two orthogonal compression axes — and where the quantization literature sits

Added 2026-09-07, after Stage 2a, before the Stage 2b compressor was written.

The KV-compression literature contains two things that are easy to conflate and must be
kept apart in this repo, because mixing them would make a Stage 2b result uninterpretable:

**Axis A — state-count compression.** `N` ordinary context positions → `M` learned memory
positions, `M ≪ N`. This changes *how many things the target attends over*, so it reduces
attention work and active context positions. C²KV, Gist/ICAE/500xCompressor, Cartridges
and LycheeMemory all live here. **This is Stage 2b.**

**Axis B — representation precision.** Each stored state goes from BF16 to INT8/INT4/INT2
or a vector-quantized code. This changes *bytes per state*, so it reduces memory footprint
and bandwidth, but the target still attends over the same number of positions.
**Everything in this section is Axis B.** It is deferred to Stage 2c and is not to be
combined with Stage 2b — a quality change would otherwise be unattributable between "the
learned state is lossy" and "the quantizer is lossy".

The four systems below all reduce **bits per entry only**; none of them reduces the number
of entries. None of these ideas is ours, and this repo has implemented none of them.

| System | Year | Axis | Mechanism | Bits reached | Needs training? |
|---|---|---|---|---|---|
| **CommVQ** | ICML 2025 | B | additive VQ + RoPE-commutative codebook, EM-learned | 2-bit solid, **1-bit with minimal loss** | codebook fit on calibration data |
| **QJL** | AAAI/ICLR 2025 | B | JL random projection then **sign** quantization | 1-bit component; 3-bit end-to-end lossless | no (data-oblivious projection) |
| **PolarQuant** | 2025 / NeurIPS 2025 | B | random preconditioning + polar transform, quantize **angles** | >4.2× compression | no |
| **QuantSpec** | 2025 | B | **hierarchical** INT4/INT8 cache, one layout serves both tiers | INT4 draft / INT8 target | no |

### 3.1 CommVQ — Commutative Vector Quantization

[arXiv:2506.18879](https://arxiv.org/abs/2506.18879) · [ICML 2025](https://proceedings.mlr.press/v267/li25du.html) · [code](https://github.com/UMass-Embodied-AGI/CommVQ)

**Additive/vector quantization.** A lightweight encoder plus a learned codebook compress
whole `d`-dimensional key and value vectors — not individual scalars — and decoding is a
single matrix multiply. Encoder and codebook are jointly fit to minimize reconstruction
MSE against the original vectors over a calibration set, using **EM**: the E-step assigns
vectors to nearest centers, the M-step updates codebook matrices by closed-form MSE
minimization.

**The RoPE-commutative construction — the part worth understanding.** RoPE is
block-diagonal, acting as independent 2×2 rotations on coordinate pairs. CommVQ constrains
each 2×2 sub-codebook `C` to satisfy `R C = C R` for every rotation block `R`. The
matrices with that property are exactly those of the form

```
[  x   y ]
[ -y   x ]
```

which is the matrix representation of multiplication by a complex number — a scaling
composed with a rotation. Such maps commute with rotation by construction.

**Why commutativity buys anything:** if the codebook commutes with RoPE, you can rotate
the *query* once and multiply it against the *quantized codes directly*, instead of
dequantizing the whole cache to full precision at every decode step. The paper reports
this drops decode cost from `O(dN_c N)` to `O(N_c N + d N_c)`. Note the asymmetry:
**the commutative constraint is needed only for keys**, because RoPE only touches the
query–key interaction. Values use ordinary additive quantization.

**Results.** 87.5% FP16 KV reduction at 2-bit while beating prior KV quantization, and
1-bit with minimal degradation — enough to run LLaMA-3.1-8B at 128K context on a single
RTX 4090. Evaluated on long-context benchmarks and GSM8K.

**What this means for *our* design — stated carefully, because the connection is easy to
oversell.** CommVQ's commutativity solves a *decoding-efficiency* problem (avoid
materializing the cache), **not** a composition problem. Our Stage 2b already gets
position-freedom for free by storing `Z` **pre-RoPE**, so we do not need commutativity to
compose chunks. Where it becomes relevant is **Stage 2c**: if we later quantize the stored
pre-RoPE state, constraining that transform to the `[[x, y], [−y, x]]` block form would
make the quantizer **RoPE-equivariant**, so re-applying position at composition time still
commutes with the codec and the two compression axes stay independent. That is a design
constraint worth inheriting later; it is not a reason to change Stage 2b.

### 3.2 QJL — 1-Bit Quantized Johnson–Lindenstrauss Transform

[arXiv:2406.03482](https://arxiv.org/abs/2406.03482) · [code](https://github.com/amirzandieh/QJL)

**Geometric preconditioning then extreme quantization.** `H_S(k) := sign(S k)` — project
onto a random subspace with a JL matrix `S`, then keep only the **sign** of each
coordinate. The JL lemma guarantees the projected inner products remain an unbiased,
low-distortion estimator of the originals, so attention scores survive.

**Why transforming before quantizing helps — the generalizable lesson.** Standard
quantizers must store a zero point and scale per block in full precision, which adds 1–2
bits per number and is pure overhead. Worse, key embeddings have **outlier coordinates**
that force a wide scale and waste the available levels. A random projection **spreads
outlier energy across all coordinates**, so the projected distribution is well-conditioned
and needs no per-channel normalization and no stored constants at all. QJL uses an
**asymmetric estimator**: quantize one side, leave the other as an unquantized JL
projection, which is unbiased with minimal distortion. Orthogonalized JL transforms improve
it further.

**Results.** 3 bits per number with no accuracy drop versus exact FP16, >5× KV memory
reduction, faster runtime, on Llama-2/Llama-3 including GQA models.

**Relevance to us:** this is the cleanest statement of *why* the deferred
Hadamard/orthogonal-rotation idea in `FUTURE_IDEAS.md` might pay off — and equally, why it
pays off **only through a quantizer**. A rotation is information-preserving; it adds
nothing on its own. It earns its keep by making a representation *quantizable*.

### 3.3 PolarQuant — polar transformation

[arXiv:2502.02617](https://arxiv.org/pdf/2502.02617) · [arXiv:2502.00527, NeurIPS 2025](https://arxiv.org/html/2502.00527) · [code](https://github.com/ericshwu/PolarQuant)

Two papers share this name and repo; they are complementary.

**Random preconditioning + polar transform.** Group coordinate pairs into 2D polar
`(radius, angle)`, then quantize the **angles**. Applied recursively `log₂ d` times — the
polar transform is re-applied to the radii — leaving one final radius and a collection of
angle vectors.

**The outlier story.** After random preconditioning the angle distribution becomes tightly
concentrated with an **analytically computable** form. That flatness is what removes
outliers and, as in QJL, eliminates the need to store per-block normalization constants.
The authors connect this explicitly to using random Hadamard matrices as preconditioners
before quantizing attention embeddings.

**A RoPE-specific observation worth remembering.** The NeurIPS paper notes that key
outliers typically appear in **only one of the two dimensions that RoPE rotates together**.
Viewed as 2D vectors those pairs show smooth, organized radius/angle structure, which
converts a channel-wise outlier problem into a well-behaved polar one. This is the same
2×2 RoPE block structure CommVQ exploits algebraically, seen from a distributional angle.

**Online suitability.** Decoding turns the query–key inner product into a **table lookup**,
with a Triton kernel for Llama and Qwen2. >4.2× compression at best-in-class quality.

### 3.4 QuantSpec — hierarchical KV, ignoring the speculative-decoding half

[arXiv:2502.10424](https://arxiv.org/abs/2502.10424) · [Apple ML Research](https://machinelearning.apple.com/research/quantspec)

**Only the cache-layout idea is in scope here. The self-speculative decoding component is
explicitly not part of Stage 2b and is not being adopted.**

**Hierarchical KV representation.** The naive way to run a low-precision draft alongside a
full-precision target is to keep *two* caches, which wastes memory and forces on-the-fly
quantization. QuantSpec instead stores **one hierarchical cache**: an INT4 tensor serves
the low-precision tier directly, and **additional residual bits** reconstruct an INT8 view
for the high-fidelity tier. Switching tiers costs no re-quantization because the low-
precision view is a prefix of the stored representation rather than a separate copy.

They also keep a small **full-precision buffer** to handle rollback correctly.

**Why it is recorded here.** The residual-tier layout — *one artifact, cheap coarse view,
optional refinement* — is structurally the same idea as the multi-fidelity page sketched
in `FUTURE_IDEAS.md` (`Z_low` + `Z_residual` + exact `C_i`). QuantSpec does it in the
bit-width dimension for a draft model; the page idea would do it across fidelity tiers for
a memory hierarchy. **The idea of a residual higher-fidelity tier is theirs, not ours**;
what would be ours, if anything, is applying it to a content-addressed context page with a
lossless raw-token tier at the bottom.

## 4. Adjacent lines worth knowing

**Position-independent caching (PIC).** The systems line that most resembles our Stage 3
control conditions. `PromptCache` (MLSys 2024) first treated the KV cache as an
independently cacheable, composable module, but reuse was constrained by a predefined
prompt schema, and quality suffered from inaccurate positional encoding and lost
cross-attention. `CacheBlend` (EuroSys 2025) fixed quality by **selectively recomputing**
a small subset of high-KV-deviation (HKVD) tokens, identified via a full first-layer
recomputation. `EPIC` recomputes only the first *k* attention-sink positions per chunk.
`CacheClip` argues CacheBlend's first-layer selection misses tokens that matter at deeper
layers, and uses a small auxiliary LLM's last-layer attention instead. `APE` uses a
shared prefix plus tuned attention temperature/scaling but cannot recover cross-chunk
attention. `ProphetKV` prioritizes tokens by query relevance.

> **Important for our Stage 7:** partial recomputation as a repair mechanism is
> well-established prior art. What differs in our brief is the *selection signal*
> (content exactness-risk: hashes, UUIDs, serials, paths, dates) and the *granularity*
> (whole page, not individual tokens). Neither is obviously better; both must be
> measured against CacheBlend-style deviation-based selection as a baseline.

**Prompt compression / soft prompts.** `Gist Tokens` (NeurIPS 2023) — up to 26× on short
prompts, but requires finetuning the base model. `AutoCompressor` (2023) — recursive,
handles 30k tokens, also finetunes. `ICAE` (2024) — frozen decoder, 4–16×, last-layer
embeddings. `500xCompressor` (ACL 2025) — frozen decoder, moves the carrier from
last-layer embeddings to **per-layer KV**, reaching 6–480× while retaining 62–73% of
capability with only 0.25% added parameters. The trend line is clear and worth
internalizing: **KV-based carriers retain detail far better than embedding-based
carriers at high ratios.** `EPL` (2025) separately criticizes ICAE/500xCompressor for
neglecting positional encoding, proposing that compression-token position IDs be
distributed uniformly across the context rather than clustered at the end — a cheap
ablation for us.

**Cartridges / self-study** ([arXiv:2506.06266](https://arxiv.org/abs/2506.06266),
[code](https://github.com/HazyResearch/cartridges)) is the most important economic
precedent. Train a small KV cache offline per corpus; amortize training cost across all
queries against that corpus. ~39× average cache memory reduction, ~26× throughput.
Two findings we must not re-discover the hard way:
1. **Naive next-token training on the corpus is not competitive with in-context
   learning.** They needed synthetic self-generated QA ("self-study") to make it work.
   This is a direct warning against the simplest possible Stage 2 objective.
2. They report that generic prompt/KV compression methods **degrade rapidly beyond ~2×**
   on hard long-context tasks. If our Stage 2 sees collapse at 4×, that is consistent
   with published results, not a bug in our harness.

`Cartridges at Scale` (2026) extends to many modular per-document cartridges that
compose, beating monolithic designs by 10–30 points, and shows cartridges combine with
retrieval to match RAG at lower cost. **This is very close to our Stages 3–6 combined.**

**Other 2026 work in the neighborhood:** `LycheeDecode` (ICLR 2026, hybrid-head sparse
decoding via retrieval heads), `LycheeCluster` (ACL 2026, hierarchical indexing of
context memory), `KV Cache Transform Coding` (ICLR 2026 — a learned codec view of KV,
relevant to the deferred RVQ/rotation ideas), `Still: Amortized KV Cache Compaction in a
Single Forward Pass` (2026), `Training Transformers for KV Cache Compressibility` (2026),
`RedKnot` (head-aware KV reuse), `QCFuse` (query-centric cache fusion), `A³`
(attention-aware cache fusion).

**Production context caching.** Anthropic prompt caching, Gemini context caching, vLLM
PagedAttention / SGLang RadixAttention / ChunkAttention. All are **exact prefix** caches:
lossless and free of quality risk, but they only fire when the reused text is a literal
prefix. The entire research area above exists because that constraint is too strong for
RAG and multi-document workloads. Our project's practical value proposition must be
measured against exact prefix caching, not only against cold native prefill.

---

## 5. What is clearly published (we must not claim these)

1. Independently prefilling chunks and composing their compressed KV at inference — **C²KV**.
2. Keeping the target model frozen and training only a small sidecar with compression
   tokens — **C²KV**, and in a different form **ICAE / 500xCompressor**.
3. Storing KV pre-RoPE and re-applying position at composition time — **C²KV**.
4. Compiling context once and amortizing across many queries — **Cartridges**.
5. Per-document modular caches that compose and combine with retrieval — **Cartridges at Scale**.
6. Block landmarks for selecting which blocks to materialize — **Landmark Attention**.
7. Chunk-wise compressed memory with a learned gate — **LycheeMemory**.
8. Selective recomputation of raw tokens to repair composed caches — **CacheBlend / EPIC / CacheClip**.
9. Content-hash-keyed dedup of cached blocks — standard practice in prefix-cache systems
   (RadixAttention). **The hash is an ID, not a contribution.**
10. Compression ratios of 4–16× with graceful degradation on semantic benchmarks — **C²KV**.

## 6. What this repo adds that is genuinely ours (all of it small)

These are *methodological* and *measurement* contributions, not architectural ones:

- **Per-fact-class loss curves.** Published results report aggregate scores on semantic
  benchmarks. We measure *what breaks first* across semantic facts vs. exact identifiers
  vs. numbers vs. dates vs. hashes vs. paths vs. URLs, with matched distractors, as a
  function of compression ratio. LycheeMemory in particular does not report this.
- **Honest cold/warm/amortized accounting.** C²KV excludes offline extraction from TTFT.
  We report cold TTFT, warm TTFT, and the full amortization curve over m ∈ {1,2,4,8,16,32}
  as a pre-registered deliverable.
- **Content-risk-routed page-level exact fallback.** CacheBlend selects tokens by KV
  deviation; we test selection by *content exactness-risk* at page granularity, with an
  oracle upper bound and a cheap non-oracle policy, and measure what fraction of pages
  ever needs exact prefill.
- **Apple Silicon unified-memory characterization.** The published latency stories are
  CUDA/HBM stories where KV transfer over PCIe is a first-order cost. On an M4 Max with
  128 GB unified memory that cost structure is different, and it is not obvious the
  reported speedups survive. This is a genuine open empirical question.
- **An aggressive control battery** (random vectors, shuffled pages, wrong-page
  retrieval, equal-token-budget raw context, independent vs. joint compilation) applied
  uniformly. Equal-token-budget raw context in particular is the control that most often
  goes missing in this literature.

## 7. Where the gaps actually are

Honest assessment of what remains unexplored after the survey above:

1. **Exactness is under-reported everywhere.** Every compressed-memory paper benchmarks
   on semantic/multi-hop QA and summarization. The hypothesis "compressed state handles
   semantics while a tiny minority of pages need exact prefill" is *assumed* by these
   systems (LycheeMemory's JIT recompression, CacheBlend's partial recomputation) but
   not **quantified as a function of content type**. This is the strongest open question.
2. **The lossless sidecar as an explicit architectural tier.** Prior systems either keep
   full KV (Landmark) or drop the raw text (C²KV, Cartridges). Treating raw tokens as a
   cheap, always-present L3 that is paged in on demand is a systems framing that is
   obvious in hindsight but not, as far as this survey found, measured end-to-end.
3. **Whether any of it matters on unified memory.** Unproven either way.
4. **The learned machine-to-machine codec** (the optional final experiment) is *partly*
   pre-empted by `KV Cache Transform Coding` (ICLR 2026) and the general codec framing.
   Do not treat it as open without re-checking that paper first.

**Consequence for planning:** Stages 1–4 are best understood as *replicating C²KV's core
claim on Apple Silicon with tighter accounting*. If replication fails, that is the
result. If it succeeds, the novel contribution lives in Stages 6–7, not Stage 3.

---

## Sources

- [C²KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference (arXiv:2607.17715)](https://arxiv.org/abs/2607.17715)
- [C²KV full text (HTML)](https://arxiv.org/html/2607.17715v1) · [code](https://github.com/s7a9/C2KV)
- [Landmark Attention: Random-Access Infinite Context Length for Transformers (arXiv:2305.16300)](https://arxiv.org/abs/2305.16300) · [code](https://github.com/epfml/landmark-attention/)
- [Dynamic Long Context Reasoning over Compressed Memory via End-to-End RL — "LycheeMemory" (arXiv:2602.08382)](https://arxiv.org/pdf/2602.08382) · [ACL Anthology](https://aclanthology.org/2026.acl-long.365/) · [org](https://github.com/LycheeMem)
- [Learning to Compress Prompts with Gist Tokens (arXiv:2304.08467)](https://arxiv.org/abs/2304.08467)
- [500xCompressor: Generalized Prompt Compression for LLMs (arXiv:2408.03094)](https://arxiv.org/html/2408.03094v1) · [ACL 2025](https://aclanthology.org/2025.acl-long.1219.pdf)
- [Prompt Compression for Large Language Models: A Survey (arXiv:2410.12388)](https://arxiv.org/html/2410.12388v2)
- [Cartridges: Lightweight and general-purpose long context representations via self-study (arXiv:2506.06266)](https://arxiv.org/abs/2506.06266) · [code](https://github.com/HazyResearch/cartridges)
- [Cartridges at Scale: Training Modular KV Caches over Large Document Collections (arXiv:2606.04557)](https://arxiv.org/html/2606.04557v1)
- [CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion (arXiv:2405.16444)](https://arxiv.org/pdf/2405.16444)
- [CacheClip (arXiv:2510.10129)](https://arxiv.org/pdf/2510.10129v1)
- [TurboRAG: Accelerating RAG with Precomputed KV Caches for Chunked Text (arXiv:2410.07590)](https://arxiv.org/pdf/2410.07590)
- [KV Prediction for Improved Time to First Token (arXiv:2410.08391)](https://arxiv.org/abs/2410.08391) · [code](https://github.com/apple/corenet/tree/main/projects/kv-prediction)
- [LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding (arXiv:2602.04541)](https://arxiv.org/html/2602.04541)
- [KV Cache Transform Coding (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/3fb6f10bd2784f6cfb6a6ed6280df40c-Paper-Conference.pdf)
- [Adaptive KV Cache Reuse for Fast Long-Context LLM Serving (arXiv:2605.24022)](https://arxiv.org/pdf/2605.24022)
- [Still: Amortized KV Cache Compaction in a Single Forward Pass (arXiv:2606.07878)](https://arxiv.org/pdf/2606.07878)
