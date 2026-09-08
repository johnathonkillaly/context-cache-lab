# Stage 3 — frozen carrier, cross-page composition

Frozen 2026-09-07 before any held-out evaluation. Stage 2b remains INCONCLUSIVE.
The user selected continuation with a frozen usable carrier, not more training.
This adds a separate experiment; EXPERIMENT.md §7 and §7b remain unchanged.

Carrier: existing Qwen/Qwen3-4B target, BF16 compute/stored KV, FP32 extractor parameters,
step-2200 checkpoint (SHA256 and model revision in results/stage3/freeze.json).
No architecture, weights, objective, compression-token, or dtype changes. No tuning
on Stage 3 quality. Primary ratio 4. Secondary 2/8/16 may run only after the primary
report exists; omission of these optional sweeps must be explicit.

## Corpus and schedule

New deterministic stage3_dev_A and stage3_test_B; original Draw B never loaded.
Each chain page contains four independently permuted relation records, with four
plausible terminal values. Removing any edge page leaves four symbolic possibilities.
Six relation templates. Each page also contains identical neutral prose (no answers,
sequence clues, or other-page identifiers). Natural page lengths are measured; they
are not padded to a hidden budget. Hops means number of relation edges/pages (2/3/4).

Held out: 24 two-page semantic items; six items per other hop/terminal cell (11 cells);
12 order tasks, each ordered and counterfactually shuffled with a changed gold answer.
Scaling and position reuse the first 12 two-page semantic items, without selecting on
performance. Scaling N=2/4/8/16/32; relevant pages first two. At N=16 also first+middle,
middle+middle, middle+last, last two. Distractor pages are a fixed nested pool.
Order pages contain one event each; only document sequence gives the successor.
No sequence numbers, timestamps, or links between events. Shuffling repositions
pre-RoPE states and changes the reference answer; unchanged query and page texts.

Development: run at most a small fixed smoke slice to validate the harness/native
ceiling. Corpus changes require a documented reason and new freeze before held-out.
Held-out byte hashes are frozen before use. One final evaluation, append-only rows,
exclusive start marker; interrupted runs may resume missing conditions only with
matching protocol, source, corpus, checkpoint, and settings hashes. No overwrites.

## Conditions and metrics

Every chain variant: NATIVE, INDEPENDENT, JOINT, BUDGET, NOCTX, RANDOM, WRONGPAGE;
NATIVE and INDEPENDENT leave each relevant page out. For two pages this explicitly
includes A, B, A+B, neither. For larger chains every edge is ablated. Distractors
remain during ablations. JOINT receives concatenated raw page tokens in the existing
single-chunk extractor; longer input is a distribution shift, reported as a limitation.
Wrong pages come from another same-class/hop item (no shared answer), are compiled
independently, and matched to each original page's token length with neutral padding
or truncation. RANDOM matches each layer's learned state mean/std. No chain shuffle.
Order tasks get all seven controls in both orders, with changed labels; missing-page
chain gate is not applied to adjacency tasks where deletion changes the target.

BUDGET retains a deterministic prefix of original raw token IDs, exactly as many
context positions as independent memory slots. Prefix selection requires no question
or gold. This refines the earlier approximate whole-chunk budget to exact positions.
All prompts use identical head/tail and question formatting; no options exposed in
questions. Report raw context tokens (available and supplied), memory positions,
active prompt positions BEFORE decoding, nominal and realized ratio, source and
supplied page counts, relevant/distractor counts and composed offsets per row.

Greedy decode, thinking disabled, max 48 tokens. Strict normalized full-answer EM,
semantic phrase correctness (case/punctuation normalization; canonical phrase plus
no competing option), and clean generation accuracy reported separately. Semantic
metric is conservative canonical-phrase matching, not an LLM judge; save predictions.
Rank margin = gold mean-token logprob minus MAX of three alternative mean logprobs.
Also save all option logprobs and rank accuracy. No per-class pooling before tables.

Primary Q is clean generation correctness. A valid chain item is selected solely
using raw controls: full NATIVE correct, NOCTX incorrect, EVERY native leave-one-out
incorrect. Report all items and valid subset, denominator and selection rate. Require
at least 8 valid primary items; otherwise task ceiling is insufficient for a verdict
about the carrier. Native full-vs-missing margins also reported. No postselection on
compiled quality. Neither single-page native condition may reach 50% correctness in aggregate;
otherwise corpus design is invalid ("usually answers" means >=50%). Four candidates
make chance guessing possible; individual valid-item selection remains stricter.

Retained Q = (compiled - NOCTX)/(NATIVE - NOCTX), undefined if denominator <=0.
Composition gain = Q(full) - max_i Q(leave-i-out), both aggregate and per-item;
rank gain uses the strongest missing-page margin per item. Unclipped values retained.

## Frozen gates at 4x

1. Valid two-page semantic native-normalized INDEPENDENT Q >=0.60.
2. At least 80% of valid two-page semantic items: full INDEPENDENT rank margin
   exceeds EVERY missing-page margin by >0.10 mean-logprob units (numerical cushion).
3. INDEPENDENT retained Q >=0.85 * JOINT retained Q. JOINT informative only if
   retained Q >=0.40; otherwise report non-informative, never a pass.
4. On the fixed scaling cohort, INDEPENDENT Q(16)/Q(2) >=0.75. Q(2)<=0 is
   non-informative. N=32 exploratory. Native curves alongside compiled curves.
5. On valid two-page semantic items, INDEPENDENT Q exceeds EACH of NOCTX, RANDOM,
   WRONGPAGE by >=0.15 absolute accuracy. This is fixed from Stage 2b's observed
   control gaps and quality scale, not calibrated to held-out results.

PASS/composable: all five gates pass and no diagnostic collapse below. PARTIAL:
gates 1/2/5 establish two-page composition but tax/scaling fails or is uninformative,
or longer-chain/order diagnostics collapse. FAIL: usable native ceiling but two-page
composition proof fails. If corpus validity/native ceiling fails, carrier verdict is
NOT ESTABLISHED, not a manufactured FAIL/PASS. Diagnostic collapse: native-normalized
semantic Q at 3/4 hops <0.5 of two-page retained Q when native Q>=0.5; or order paired
success <0.5 of native paired success when native paired success>=0.5. Non-informative
native ceilings explicitly bound conclusions, not silently removed from the report.

## Timing and numerical QA

Per-condition synchronized diagnostic timing: compile each page, RoPE reassignment,
concatenation/cache setup, target question-tail processing, first-token argmax.
Compiled pages are K/V cache input, with zero target forward tokens over original
pages. Warm includes head prefill plus composition + tail + first token; cold adds
compilation. Serialized state load is N/A (in-memory reuse), never zero-cost disk IO.
A representative dev item gets 2 warmups and 5 trials, all samples and medians saved.
No speedup claim or timing gate. Quality-run single timings are diagnostics only.

Tests: capture pre-RoPE keys, exactly-once fresh positioning, offset dependence,
immutable independent state, AB versus BA positioned keys, no double rotation,
timed compose equivalence, and existing live ratio-1 identity tolerance (max logit
error <2.0 plus same argmax). Ratio-1 is positional identity, not a claim that
independent native chunk prefills equal joint native context at deeper layers.

Final report answers the user's 12 questions, with controls, per-class/hop tables,
order counterfactual pairs, scaling/position curves, gate JSON, uncertainty and small-n
limitations. Axis B, retrieval/Landmark, exact fallback and all other extensions stay
unimplemented. Only clean Stage 3 evidence can justify proposing quantization.
