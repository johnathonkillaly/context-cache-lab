# Stage 3: FAIL

**Stage 2b remains INCONCLUSIVE.** Frozen step-2200 full-precision carrier; primary ratio 4×. Original Draw B untouched. Secondary ratios not run.

Held-out measurements: 2358 rows. Native-valid primary items: 13/24. No compiled-score filtering.

INDEPENDENT clean-accuracy Wilson 95% interval on valid items: [0, 0.22810184305529166]. Small samples, particularly n=6 terminal/hop cells, limit precision.

## Frozen criteria

| Criterion | Result | Evidence |
|---|---|---|
| 1_retained_quality | MISS | {"value": 0.0, "threshold": 0.6} |
| 2_missing_page_dependence | MISS | {"value": 0.07692307692307693, "threshold": 0.8} |
| 3_independent_tax | NON-INFORMATIVE | {"independent_retained": 0.0, "joint_retained": 0.0, "threshold": 0.85, "informative": false} |
| 4_scaling | NON-INFORMATIVE | {"q2": 0.0, "q16": 0.0, "retention": null, "threshold": 0.75, "informative": false} |
| 5_controls | MISS | {"gaps": {"NOCTX": 0.0, "RANDOM": 0.0, "WRONGPAGE": 0.0}, "threshold": 0.15} |

## Terminal classes and chain depth

Q is clean generation correctness. EM requires the complete normalized answer; semantic correctness uses canonical phrase matching and excludes competing options. Rank margin is against the strongest of three alternatives. Options never appear in the question.

| Hops / terminal | n / valid | Native | Independent | Joint | Budget | No context | Random | Wrong page | Retained | Compose gain |
|---|---|---|---|---|---|---|---|---|---|---|
| 2_semantic | 24 / 13 | 0.792 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2_identifier | 6 / 1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2_number | 6 / 6 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2_hash | 6 / 4 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3_semantic | 6 / 0 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3_identifier | 6 / 3 | 0.833 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3_number | 6 / 3 | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3_hash | 6 / 1 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4_semantic | 6 / 0 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4_identifier | 6 / 0 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4_number | 6 / 1 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4_hash | 6 / 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | N/A | 0.000 |

| Hops / terminal | Independent EM | Semantic correctness | Rank accuracy | Native margin | Independent margin | Mean per-item rank gain |
|---|---|---|---|---|---|---|
| 2_semantic | 0.000 | 0.000 | 0.375 | 2.648 | -1.881 | -1.474 |
| 2_identifier | 0.000 | 0.000 | 0.333 | 3.134 | -1.218 | -1.286 |
| 2_number | 0.000 | 0.000 | 0.000 | 4.518 | -1.037 | -0.965 |
| 2_hash | 0.000 | 0.000 | 0.667 | 1.944 | 0.014 | -0.371 |
| 3_semantic | 0.000 | 0.000 | 0.333 | -0.873 | -2.036 | -0.940 |
| 3_identifier | 0.000 | 0.000 | 0.000 | 1.850 | -0.842 | -0.238 |
| 3_number | 0.000 | 0.000 | 0.167 | 1.147 | -1.203 | -1.646 |
| 3_hash | 0.000 | 0.000 | 0.000 | 0.372 | -1.179 | -0.981 |
| 4_semantic | 0.000 | 0.000 | 0.167 | -2.847 | -3.362 | -2.110 |
| 4_identifier | 0.000 | 0.000 | 0.500 | 0.032 | -0.373 | -0.532 |
| 4_number | 0.000 | 0.000 | 0.333 | -1.070 | 0.024 | -0.273 |
| 4_hash | 0.000 | 0.000 | 0.167 | -0.988 | -0.984 | -0.530 |

## Order counterfactuals

Paired success requires BOTH original and shuffled (relabelled) answers correct. Each page is unchanged; only page order changes. Interpret compressed order only where native succeeds.

| Condition | Ordered | Shuffled, new gold | Paired success |
|---|---|---|---|
| NATIVE | 1.000 | 1.000 | 1.000 |
| NOCTX | 0.000 | 0.000 | 0.000 |
| INDEPENDENT | 0.000 | 0.000 | 0.000 |
| JOINT | 0.000 | 0.000 | 0.000 |
| BUDGET | 0.000 | 0.000 | 0.000 |
| RANDOM | 0.000 | 0.000 | 0.000 |
| WRONGPAGE | 0.000 | 0.000 | 0.000 |

## Page count and placement

Same 12 examples, two relevant pages; fixed nested distractors. Page-count sweep keeps relevant pages first. Position sweep holds N=16; base is N=2. No selection on quality.

| Variant | Native | Independent | Joint | Budget | NOCTX | RANDOM | WRONGPAGE |
|---|---|---|---|---|---|---|---|
| base | 0.833 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pages_4 | 0.583 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pages_8 | 0.500 | 0.000 | 0.000 | 0.833 | 0.000 | 0.000 | 0.000 |
| pages_16 | 0.750 | 0.000 | 0.000 | 0.667 | 0.000 | 0.000 | 0.000 |
| pages_32 | 0.583 | 0.000 | 0.000 | 0.667 | 0.000 | 0.000 | 0.000 |
| first_middle | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| middle_middle | 0.583 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| middle_last | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| last_two | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Accounting and timing

Every JSONL row includes supplied/available raw tokens, memory positions, pre-decode active prompt positions, nominal/realized ratio, page counts, offsets, and synchronized component timings. BUDGET is an exact token-prefix budget matched to independent memory slots. Missing-page runs retain distractors. JOINT is the existing whole-document extractor and can be out of its training length distribution. Serialized-state loading is N/A; states are reused in memory. Single quality-run times are diagnostic, not timing medians. See timing_dev.json for 2-warmup/5-trial timing QA. No speedup is claimed.

## Interpretation

The frozen carrier fails this new cross-page task distribution. The failure already appears with two relevant pages; JOINT also fails, so neither an independence tax nor a page-selection remedy is established. Detailed answers and limitations follow. No quantization, retrieval, fallback, or further training was implemented.

## Primary composition proof and budgets

NATIVE solves 19/24 two-page semantic items. The terminal page alone guesses 6/24 correctly; the first page alone solves 0/24. Excluding those guesses and the five native failures leaves 13 valid items. These exclusions use raw controls only. All four symbolic terminal possibilities remain possible when an edge page is removed.

On those 13 items, INDEPENDENT, JOINT, BUDGET, NOCTX, RANDOM, WRONGPAGE, and both compiled single-page conditions all have zero clean generation accuracy. Thus the retained-quality gate is 0/13, not a close miss. The Wilson 95% interval for independent accuracy is 0–22.8%, below the frozen 60% bar. Rank-margin dependence succeeds on 1/13 items (7.7%; Wilson 95% interval 1.4–33.3%), below the frozen 80% bar. Both full-versus-missing generation gains are zero.

Mean counts below are over the 13 valid primary items. Available raw context averages 196.23 tokens, on two relevant pages with no unrelated pages. Exact per-row counts and supplied page indices are in the JSONL. Full conditions use nominal 4× compression; NATIVE is the uncompressed reference. LOO 0 retains only page B; LOO 1 retains only page A.

| Condition | Clean Q | Rank accuracy | Rank margin | Raw tokens supplied | Memory positions | Active prompt positions |
|---|---|---|---|---|---|---|
| NATIVE | 1.000 | 1.000 | 3.960 | 196.231 | 0.000 | 293.769 |
| NOCTX | 0.000 | 0.462 | -1.103 | 0.000 | 0.000 | 93.538 |
| INDEPENDENT | 0.000 | 0.308 | -2.474 | 0.000 | 49.692 | 147.231 |
| JOINT | 0.000 | 0.462 | -1.493 | 0.000 | 49.308 | 146.846 |
| BUDGET | 0.000 | 0.308 | -1.187 | 49.692 | 0.000 | 147.231 |
| RANDOM | 0.000 | 0.462 | -0.763 | 0.000 | 49.692 | 147.231 |
| WRONGPAGE | 0.000 | 0.231 | -2.240 | 0.000 | 49.692 | 147.231 |
| NATIVE_LOO_0 | 0.000 | 0.231 | -2.175 | 102.769 | 0.000 | 200.308 |
| NATIVE_LOO_1 | 0.000 | 0.385 | -1.311 | 93.462 | 0.000 | 191.000 |
| INDEPENDENT_LOO_0 | 0.000 | 0.308 | -2.554 | 0.000 | 26.077 | 123.615 |
| INDEPENDENT_LOO_1 | 0.000 | 0.462 | -0.788 | 0.000 | 23.615 | 121.154 |

INDEPENDENT’s mean rank margin is worse than NOCTX on the valid subset. Occasional correct option rankings are not sufficient evidence of composition: NOCTX and RANDOM also rank some options correctly. WRONGPAGE supplies unrelated states with the same shape, and its relevant-page count is zero. JOINT has slightly fewer memory slots because ceiling division is applied once rather than once per page; this small accounting difference is explicit, not hidden.

## Interference and placement below the generation floor

All independent generation scores are zero, so Q(16)/Q(2) is undefined. Flat zero accuracy is not low interference. The rank-margin curves do not show a monotonic loss as pages accumulate, but are negative throughout and do not establish useful stability. The raw-prefix BUDGET eventually includes both early relevant pages as its absolute budget grows; its higher scores at N≥8 are therefore expected from selection. This does not rescue compressed composition.

| Layout | Native margin | Independent margin | Joint margin | NOCTX margin | RANDOM margin | WRONGPAGE margin |
|---|---|---|---|---|---|---|
| base | 2.353 | -2.566 | -1.960 | -1.777 | -1.462 | -3.256 |
| pages_4 | 0.891 | -2.492 | -2.127 | -1.777 | -1.670 | -3.017 |
| pages_8 | 0.514 | -2.308 | -1.982 | -1.777 | -1.875 | -2.967 |
| pages_16 | 1.662 | -2.400 | -2.015 | -1.777 | -1.898 | -2.802 |
| pages_32 | 0.927 | -2.425 | -2.151 | -1.777 | -1.779 | -2.529 |
| first_middle | -1.562 | -2.382 | -2.343 | -1.777 | -1.974 | -2.791 |
| middle_middle | 0.409 | -2.397 | -2.461 | -1.777 | -1.906 | -2.800 |
| middle_last | 0.074 | -2.321 | -2.509 | -1.777 | -1.779 | -2.778 |
| last_two | 1.402 | -2.301 | -2.607 | -1.777 | -1.709 | -2.802 |

## Representative timing and execution record

Development item `stage3_dev_A_semantic_2_000`: 199 raw context tokens → 50 memory positions. Median of five synchronized trials after two warmups; all samples retained in `timing_dev.json`. These short-page measurements are not comparable to Stage 2b’s 4K headline timings.

| Component | Median seconds |
|---|---|
| Page compilation total | 0.310455 |
| Prompt-head prefill | 0.075754 |
| RoPE reassignment | 0.002887 |
| State concatenation | 0.000875 |
| Cache setup | 0.001319 |
| Target question/tail processing | 0.079206 |
| First-token selection | 0.000358 |
| Warm TTFT | 0.160448 |
| Cold TTFT | 0.470903 |
| Native TTFT | 0.243606 |

Serialized-state loading is not applicable: this run reuses in-memory states. The target forwards zero original-page tokens on the compiled query path. First-token selection is argmax from the final prompt logits; no extra decode forward is required for that token. No speedup or timing success claim is made.

The single held-out run saved 2,358 rows across 210 variants, from 2026-09-08T01:18:26.135639+00:00 to 2026-09-08T02:17:36.002565+00:00 (about 59 minutes). It completed without retrying conditions. Each row records model revision, checkpoint hash, seed, draw, commit and timestamp. The `-dirty` commit suffix includes newly written run artifacts; the frozen evaluation source hashes provide exact code identity.

## What the evidence answers

1. **Do independent pages support genuine two-page reasoning?** Not at the preregistered useful-quality level for this frozen carrier. It gets 0/13 valid semantic items correct, against NATIVE 13/13 and all three negative controls 0/13. This Stage 3 result is FAIL; the historical Stage 2b verdict remains INCONCLUSIVE.
2. **Does every required page contribute?** Not reliably. Only 1/13 valid items has a full-state rank margin more than 0.10 above every missing-page margin. Generation composition gain is zero. The symbolic graph requires both pages, but the compiled model does not reliably exploit that dependence.
3. **How far do chains extend?** No successful compiled depth is established: generation is zero at 2, 3 and 4 hops for every terminal class. Native semantic accuracy also falls from 19/24 at two hops to 2/6 at three and 1/6 at four; there are no native-valid semantic items at three or four hops. Therefore this run cannot locate an additional compressed-only depth limit beyond the two-page failure.
4. **What is the independence tax?** Not identifiable on the primary quality metric: INDEPENDENT and JOINT both score zero. JOINT fails the frozen informativeness floor, so equality is not a pass. A diagnostic mean-margin difference of −0.982 favors JOINT on valid primary items, but both are negative; this is not a usable-quality tax estimate. No high-ratio comparison or regularization explanation is inferred.
5. **Does order survive?** The operational order task fails for compiled states: NATIVE solves all 12 ordered/shuffled pairs, INDEPENDENT and JOINT solve none. Independent option ranking is also 0/24 across both orders (native 24/24). Positioning tests pass, so mechanically applying new positions is insufficient to establish semantic order use. Because content recall may also fail, this does not isolate an order-only representation defect.
6. **What happens as irrelevant pages accumulate?** Independent generation is zero from N=2 through N=32. The 2→16 retention fraction is undefined. Native accuracy is 0.833/0.583/0.500/0.750/0.583 across the five counts. Negative, nearly flat compiled rank margins cannot establish useful low-interference memory; neither do these data establish strong compressed-page interference.
7. **Does relevant-page position matter?** Native Q at N=16 varies from 0.333 to 0.750 across placements. Independent Q remains zero, with mean margins from −2.400 to −2.301. The numerical offset tests confirm keys change correctly, but the quality floor prevents a meaningful placement-robustness claim.
8. **Do semantics survive better than exact terminals?** Not in generated answers on this corpus: every compiled class is at zero. Option-ranking results vary, including 4/6 correct two-hop hash rankings, but the sample is tiny and missing-page controls do not establish usable composition. This experiment does not reproduce a semantic-over-exact generation advantage; it does not contradict the preserved Stage 2b measurements on its different corpus.
9. **Is the next bottleneck fidelity or page selection?** The failure exists with just two relevant pages and no unrelated pages, and JOINT also fails. Carrier/task-transfer fidelity is the immediate unresolved issue. This design cannot fully separate loss of within-page relations from failure to combine intact relations: it lacks a separate direct page-local query assay on these new pages. That limitation prevents attributing the entire failure specifically to independent composition.
10. **Is Axis B quantization justified?** No. The carrier failed the useful-composition gate, so another lossy stage is not justified. No quantization or further training was run.
11. **Is Landmark/retrieval justified?** This run does not establish it as the remedy for compiled memory. Selection cannot remove distractors from the already failing two-relevant-page case. Native/BUDGET differences are descriptive evidence about raw-context selection, not a validated compressed-page selection proposal. No retrieval was implemented.
12. **Does this support the eventual content-addressed page cache?** Pre-RoPE storage, deterministic independent compilation, immutable reuse, and positional reassignment remain mechanically valid. The evidence does not support treating this frozen carrier as general composable memory. Preserve the negative result and stop optimization of that claim; any further diagnostic should be a separately preregistered experiment with a fresh draw, not a retest or relabeling of Stage 3.

## Limits and verification

- Primary validity is 13/24 and uses native controls only. Some secondary diagnostics have very few valid items: two-hop identifiers have only 1/6 (the terminal page alone often guesses correctly), and several longer-chain cells have none. Those cells are not independent evidence of composition and would need redesign on a new draw before a future confirmatory test.
- New pages are naturally short (roughly 94–100 tokens per page on the valid semantic cohort), branching and dense in relation records; Stage 2b training used roughly 256-token chunks from a different corpus. Format, page length, task and terminal style all change together. Thus the failure establishes insufficient transfer/usefulness of this frozen carrier here, not the impossibility of all independently compiled memory architectures.
- Semantic correctness is a conservative canonical-phrase metric, not an external semantic judge. Inspection of all 24 primary independent predictions found substitutions, missing attributes and repetition rather than clean paraphrases of the gold phrase. Rank metrics independently fail to show the required composition dependence. Predictions remain available for auditing; no human relabeling changed the gate.
- 101 fast tests pass. Two live RoPE tests passed before the final run, including the historical ratio-1 tolerance (same argmax; max logit error <2.0). The new extractor hook check verifies pre-RoPE storage and immutable deterministic reuse. After the final run, a development-only comparison against the unchanged historical Stage 2b composed-answer helper matched prediction, rank margin and gold logprob exactly (`historical_path_qa.json`). No held-out measurement was repeated.
- `check_stage3_results.py` verifies complete condition coverage, exact BUDGET/INDEPENDENT active-position equality, negative-control shape matching, finite margins, no duplicate rows, source/corpus/checkpoint hashes and historical raw-result hashes. Git also confirms the historical carrier code and Stage 1–2 result directories have no changes.
- The primary protocol, five gates and evaluation sources were frozen before held-out use. The freeze records an earlier pre-held-out clarification of the corpus-level “usually answers” cutoff. Nothing was calibrated against held-out quality. Secondary 2×/8×/16× sweeps remain unrun; they would not convert this failed 4× gate into a pass.

Artifacts: `stage3_gate.json` (gate calculations and native-valid IDs), `condition_summary.csv` (grouped metric/accounting means), `stage3_test_B_r4.jsonl` (every prediction, option score, count and timing), `.run.json`/`.complete.json` (execution manifest and integrity), `freeze.json` (carrier/corpus/source identities). Historical state is tagged `pre-stage3-frozen`.
