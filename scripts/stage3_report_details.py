#!/usr/bin/env python
"""Reproducible descriptive tables and interpretation of the completed frozen run.
No model evaluation and no changes to frozen gate computation.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import csv
import json
import statistics
from collections import defaultdict
from ccl.stage3_eval import OUT,sha


def avg(rows,key): return statistics.mean(float(r[key]) for r in rows)


def details(rows,gate):
    valid=set(gate['validity']['valid_ids'])
    primary=[r for r in rows if r['item_id'] in valid and r['variant']=='base']
    lines=['## Primary composition proof and budgets','',
        'NATIVE solves 19/24 two-page semantic items. The terminal page alone guesses 6/24 correctly; the first page alone solves 0/24. Excluding those guesses and the five native failures leaves 13 valid items. These exclusions use raw controls only. All four symbolic terminal possibilities remain possible when an edge page is removed.',
        '', 'On those 13 items, INDEPENDENT, JOINT, BUDGET, NOCTX, RANDOM, WRONGPAGE, and both compiled single-page conditions all have zero clean generation accuracy. Thus the retained-quality gate is 0/13, not a close miss. The Wilson 95% interval for independent accuracy is 0–22.8%, below the frozen 60% bar. Rank-margin dependence succeeds on 1/13 items (7.7%; Wilson 95% interval 1.4–33.3%), below the frozen 80% bar. Both full-versus-missing generation gains are zero.',
        '', 'Mean counts below are over the 13 valid primary items. Available raw context averages 196.23 tokens, on two relevant pages with no unrelated pages. Exact per-row counts and supplied page indices are in the JSONL. Full conditions use nominal 4× compression; NATIVE is the uncompressed reference. LOO 0 retains only page B; LOO 1 retains only page A.',
        '', '| Condition | Clean Q | Rank accuracy | Rank margin | Raw tokens supplied | Memory positions | Active prompt positions |',
        '|---|---|---|---|---|---|---|']
    for c in ['NATIVE','NOCTX','INDEPENDENT','JOINT','BUDGET','RANDOM','WRONGPAGE','NATIVE_LOO_0','NATIVE_LOO_1','INDEPENDENT_LOO_0','INDEPENDENT_LOO_1']:
        s=[r for r in primary if r['condition']==c]
        lines.append(f'| {c} | '+' | '.join(f'{avg(s,k):.3f}' for k in ['clean_hit','rank_correct','rank_margin','raw_context_tokens_supplied','compiled_state_positions','active_prompt_positions'])+' |')
    lines += ['', 'INDEPENDENT’s mean rank margin is worse than NOCTX on the valid subset. Occasional correct option rankings are not sufficient evidence of composition: NOCTX and RANDOM also rank some options correctly. WRONGPAGE supplies unrelated states with the same shape, and its relevant-page count is zero. JOINT has slightly fewer memory slots because ceiling division is applied once rather than once per page; this small accounting difference is explicit, not hidden.',
        '', '## Interference and placement below the generation floor','',
        'All independent generation scores are zero, so Q(16)/Q(2) is undefined. Flat zero accuracy is not low interference. The rank-margin curves do not show a monotonic loss as pages accumulate, but are negative throughout and do not establish useful stability. The raw-prefix BUDGET eventually includes both early relevant pages as its absolute budget grows; its higher scores at N≥8 are therefore expected from selection. This does not rescue compressed composition.',
        '', '| Layout | Native margin | Independent margin | Joint margin | NOCTX margin | RANDOM margin | WRONGPAGE margin |',
        '|---|---|---|---|---|---|---|']
    for name,s in gate['scaling_position'].items():
        lines.append(f'| {name} | '+' | '.join(f"{s[c]['margin']:.3f}" for c in ['NATIVE','INDEPENDENT','JOINT','NOCTX','RANDOM','WRONGPAGE'])+' |')
    lines += ['', '## Representative timing and execution record','']
    t=json.loads((OUT/'timing_dev.json').read_text()); med=t['medians']['independent']; native=t['medians']['native']
    lines += [f"Development item `{t['item_id']}`: {t['raw_tokens']} raw context tokens → {t['memory_positions']} memory positions. Median of five synchronized trials after two warmups; all samples retained in `timing_dev.json`. These short-page measurements are not comparable to Stage 2b’s 4K headline timings.",
        '', '| Component | Median seconds |','|---|---|']
    for name,k in [('Page compilation total','compile_s'),('Prompt-head prefill','head_prefill_s'),('RoPE reassignment','rope_s'),('State concatenation','concat_s'),('Cache setup','cache_setup_s'),('Target question/tail processing','target_tail_s'),('First-token selection','first_token_s'),('Warm TTFT','warm_ttft_s'),('Cold TTFT','cold_ttft_s')]:
        lines.append(f'| {name} | {med[k]:.6f} |')
    lines += [f"| Native TTFT | {native['warm_ttft_s']:.6f} |",'',
        'Serialized-state loading is not applicable: this run reuses in-memory states. The target forwards zero original-page tokens on the compiled query path. First-token selection is argmax from the final prompt logits; no extra decode forward is required for that token. No speedup or timing success claim is made.',
        '',f"The single held-out run saved {len(rows):,} rows across 210 variants, from {rows[0]['timestamp_utc']} to {rows[-1]['timestamp_utc']} (about 59 minutes). It completed without retrying conditions. Each row records model revision, checkpoint hash, seed, draw, commit and timestamp. The `-dirty` commit suffix includes newly written run artifacts; the frozen evaluation source hashes provide exact code identity.",
        '', '## What the evidence answers','',
        '1. **Do independent pages support genuine two-page reasoning?** Not at the preregistered useful-quality level for this frozen carrier. It gets 0/13 valid semantic items correct, against NATIVE 13/13 and all three negative controls 0/13. This Stage 3 result is FAIL; the historical Stage 2b verdict remains INCONCLUSIVE.',
        '2. **Does every required page contribute?** Not reliably. Only 1/13 valid items has a full-state rank margin more than 0.10 above every missing-page margin. Generation composition gain is zero. The symbolic graph requires both pages, but the compiled model does not reliably exploit that dependence.',
        '3. **How far do chains extend?** No successful compiled depth is established: generation is zero at 2, 3 and 4 hops for every terminal class. Native semantic accuracy also falls from 19/24 at two hops to 2/6 at three and 1/6 at four; there are no native-valid semantic items at three or four hops. Therefore this run cannot locate an additional compressed-only depth limit beyond the two-page failure.',
        '4. **What is the independence tax?** Not identifiable on the primary quality metric: INDEPENDENT and JOINT both score zero. JOINT fails the frozen informativeness floor, so equality is not a pass. A diagnostic mean-margin difference of −0.982 favors JOINT on valid primary items, but both are negative; this is not a usable-quality tax estimate. No high-ratio comparison or regularization explanation is inferred.',
        '5. **Does order survive?** The operational order task fails for compiled states: NATIVE solves all 12 ordered/shuffled pairs, INDEPENDENT and JOINT solve none. Independent option ranking is also 0/24 across both orders (native 24/24). Positioning tests pass, so mechanically applying new positions is insufficient to establish semantic order use. Because content recall may also fail, this does not isolate an order-only representation defect.',
        '6. **What happens as irrelevant pages accumulate?** Independent generation is zero from N=2 through N=32. The 2→16 retention fraction is undefined. Native accuracy is 0.833/0.583/0.500/0.750/0.583 across the five counts. Negative, nearly flat compiled rank margins cannot establish useful low-interference memory; neither do these data establish strong compressed-page interference.',
        '7. **Does relevant-page position matter?** Native Q at N=16 varies from 0.333 to 0.750 across placements. Independent Q remains zero, with mean margins from −2.400 to −2.301. The numerical offset tests confirm keys change correctly, but the quality floor prevents a meaningful placement-robustness claim.',
        '8. **Do semantics survive better than exact terminals?** Not in generated answers on this corpus: every compiled class is at zero. Option-ranking results vary, including 4/6 correct two-hop hash rankings, but the sample is tiny and missing-page controls do not establish usable composition. This experiment does not reproduce a semantic-over-exact generation advantage; it does not contradict the preserved Stage 2b measurements on its different corpus.',
        '9. **Is the next bottleneck fidelity or page selection?** The failure exists with just two relevant pages and no unrelated pages, and JOINT also fails. Carrier/task-transfer fidelity is the immediate unresolved issue. This design cannot fully separate loss of within-page relations from failure to combine intact relations: it lacks a separate direct page-local query assay on these new pages. That limitation prevents attributing the entire failure specifically to independent composition.',
        '10. **Is Axis B quantization justified?** No. The carrier failed the useful-composition gate, so another lossy stage is not justified. No quantization or further training was run.',
        '11. **Is Landmark/retrieval justified?** This run does not establish it as the remedy for compiled memory. Selection cannot remove distractors from the already failing two-relevant-page case. Native/BUDGET differences are descriptive evidence about raw-context selection, not a validated compressed-page selection proposal. No retrieval was implemented.',
        '12. **Does this support the eventual content-addressed page cache?** Pre-RoPE storage, deterministic independent compilation, immutable reuse, and positional reassignment remain mechanically valid. The evidence does not support treating this frozen carrier as general composable memory. Preserve the negative result and stop optimization of that claim; any further diagnostic should be a separately preregistered experiment with a fresh draw, not a retest or relabeling of Stage 3.',
        '', '## Limits and verification','',
        '- Primary validity is 13/24 and uses native controls only. Some secondary diagnostics have very few valid items: two-hop identifiers have only 1/6 (the terminal page alone often guesses correctly), and several longer-chain cells have none. Those cells are not independent evidence of composition and would need redesign on a new draw before a future confirmatory test.',
        '- New pages are naturally short (roughly 94–100 tokens per page on the valid semantic cohort), branching and dense in relation records; Stage 2b training used roughly 256-token chunks from a different corpus. Format, page length, task and terminal style all change together. Thus the failure establishes insufficient transfer/usefulness of this frozen carrier here, not the impossibility of all independently compiled memory architectures.',
        '- Semantic correctness is a conservative canonical-phrase metric, not an external semantic judge. Inspection of all 24 primary independent predictions found substitutions, missing attributes and repetition rather than clean paraphrases of the gold phrase. Rank metrics independently fail to show the required composition dependence. Predictions remain available for auditing; no human relabeling changed the gate.',
        '- 101 fast tests pass. Two live RoPE tests passed before the final run, including the historical ratio-1 tolerance (same argmax; max logit error <2.0). The new extractor hook check verifies pre-RoPE storage and immutable deterministic reuse. After the final run, a development-only comparison against the unchanged historical Stage 2b composed-answer helper matched prediction, rank margin and gold logprob exactly (`historical_path_qa.json`). No held-out measurement was repeated.',
        '- `check_stage3_results.py` verifies complete condition coverage, exact BUDGET/INDEPENDENT active-position equality, negative-control shape matching, finite margins, no duplicate rows, source/corpus/checkpoint hashes and historical raw-result hashes. Git also confirms the historical carrier code and Stage 1–2 result directories have no changes.',
        '- The primary protocol, five gates and evaluation sources were frozen before held-out use. The freeze records an earlier pre-held-out clarification of the corpus-level “usually answers” cutoff. Nothing was calibrated against held-out quality. Secondary 2×/8×/16× sweeps remain unrun; they would not convert this failed 4× gate into a pass.',
        '', 'Artifacts: `stage3_gate.json` (gate calculations and native-valid IDs), `condition_summary.csv` (grouped metric/accounting means), `stage3_test_B_r4.jsonl` (every prediction, option score, count and timing), `.run.json`/`.complete.json` (execution manifest and integrity), `freeze.json` (carrier/corpus/source identities). Historical state is tagged `pre-stage3-frozen`.']
    return '\n'.join(lines)+'\n'


def write_summary(rows):
    group=defaultdict(list)
    dims=['task','hops','terminal','variant','total_pages','relevant_pages','distractor_pages','condition','ratio']
    metrics=['clean_hit','exact_match','semantic_correct','rank_correct','rank_margin','raw_context_tokens_available','raw_context_tokens_supplied','compiled_state_positions','active_prompt_positions','supplied_pages','supplied_relevant_pages','supplied_distractor_pages']
    for r in rows: group[tuple(r[k] for k in dims)].append(r)
    with (OUT/'condition_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=dims+['n']+metrics,lineterminator='\n');w.writeheader()
        for key,rs in sorted(group.items()):
            w.writerow(dict(zip(dims,key))|{'n':len(rs)}|{k:avg(rs,k) for k in metrics})


if __name__=='__main__':
    path=OUT/'stage3_test_B_r4.jsonl'
    assert sha(path)==json.loads(path.with_suffix('.complete.json').read_text())['sha256']
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    gate=json.loads((OUT/'stage3_gate.json').read_text())
    write_summary(rows)
    report=OUT/'stage3_report.md'
    s=report.read_text().split('## Primary composition proof and budgets')[0]
    s=s.replace('See the accompanying final interpretation below; frozen gate JSON contains all denominators and native-only validity IDs. No quantization, retrieval, fallback, or further training was implemented.',
        'The frozen carrier fails this new cross-page task distribution. The failure already appears with two relevant pages; JOINT also fails, so neither an independence tax nor a page-selection remedy is established. Detailed answers and limitations follow. No quantization, retrieval, fallback, or further training was implemented.')
    report.write_text(s.rstrip()+'\n\n'+details(rows,gate))
    print(f'Wrote detailed report and condition_summary.csv')
