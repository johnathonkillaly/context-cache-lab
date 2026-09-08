#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
from ccl.stage3_eval import OUT,sha
from ccl.stage3_report import analyze

def fmt(v): return 'N/A' if v is None else f'{v:.3f}'

if __name__=='__main__':
    path=OUT/'stage3_test_B_r4.jsonl'
    complete=json.loads(path.with_suffix('.complete.json').read_text())
    assert sha(path)==complete['sha256']
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    result=analyze(rows)
    (OUT/'stage3_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Stage 3: '+result['verdict'],'',
           '**Stage 2b remains INCONCLUSIVE.** Frozen step-2200 full-precision carrier; primary ratio 4×. Original Draw B untouched. Secondary ratios not run.',
           '',f"Held-out measurements: {len(rows)} rows. Native-valid primary items: {result['validity']['valid_n']}/{result['validity']['primary_n']}. No compiled-score filtering.",
           '',f"INDEPENDENT clean-accuracy Wilson 95% interval on valid items: {result['primary_valid_wilson95']}. Small samples, particularly n=6 terminal/hop cells, limit precision.",
           '', '## Frozen criteria','', '| Criterion | Result | Evidence |','|---|---|---|']
    for name,g in result['gates'].items(): lines.append(f"| {name} | {'PASS' if g['pass'] else 'NON-INFORMATIVE' if g.get('informative') is False else 'MISS'} | {json.dumps({k:v for k,v in g.items() if k!='pass'})} |")
    lines += ['','## Terminal classes and chain depth','', 'Q is clean generation correctness. EM requires the complete normalized answer; semantic correctness uses canonical phrase matching and excludes competing options. Rank margin is against the strongest of three alternatives. Options never appear in the question.','', '| Hops / terminal | n / valid | Native | Independent | Joint | Budget | No context | Random | Wrong page | Retained | Compose gain |','|---|---|---|---|---|---|---|---|---|---|---|']
    for k,c in result['cells'].items(): lines.append(f"| {k} | {c['n']} / {c['valid_n']} | "+' | '.join(fmt(c[x]['q']) for x in ['NATIVE','INDEPENDENT','JOINT','BUDGET','NOCTX','RANDOM','WRONGPAGE'])+f" | {fmt(c['retained'])} | {fmt(c['composition_gain'])} |")
    lines += ['','| Hops / terminal | Independent EM | Semantic correctness | Rank accuracy | Native margin | Independent margin | Mean per-item rank gain |','|---|---|---|---|---|---|---|']
    for k,c in result['cells'].items(): lines.append(f"| {k} | "+' | '.join(fmt(v) for v in [c['INDEPENDENT']['exact_match'],c['INDEPENDENT']['semantic_correct'],c['INDEPENDENT']['rank_accuracy'],c['NATIVE']['margin'],c['INDEPENDENT']['margin'],c['mean_item_rank_gain']])+' |')
    lines += ['','## Order counterfactuals','', 'Paired success requires BOTH original and shuffled (relabelled) answers correct. Each page is unchanged; only page order changes. Interpret compressed order only where native succeeds.','', '| Condition | Ordered | Shuffled, new gold | Paired success |','|---|---|---|---|']
    for c,v in result['order'].items(): lines.append(f"| {c} | {fmt(v['ordered'])} | {fmt(v['shuffled_relabelled'])} | {fmt(v['paired'])} |")
    lines += ['','## Page count and placement','', 'Same 12 examples, two relevant pages; fixed nested distractors. Page-count sweep keeps relevant pages first. Position sweep holds N=16; base is N=2. No selection on quality.','', '| Variant | Native | Independent | Joint | Budget | NOCTX | RANDOM | WRONGPAGE |','|---|---|---|---|---|---|---|---|']
    for label,s in result['scaling_position'].items(): lines.append(f'| {label} | '+' | '.join(fmt(s[c]['q']) for c in ['NATIVE','INDEPENDENT','JOINT','BUDGET','NOCTX','RANDOM','WRONGPAGE'])+' |')
    lines += ['','## Accounting and timing','', 'Every JSONL row includes supplied/available raw tokens, memory positions, pre-decode active prompt positions, nominal/realized ratio, page counts, offsets, and synchronized component timings. BUDGET is an exact token-prefix budget matched to independent memory slots. Missing-page runs retain distractors. JOINT is the existing whole-document extractor and can be out of its training length distribution. Serialized-state loading is N/A; states are reused in memory. Single quality-run times are diagnostic, not timing medians. See timing_dev.json for 2-warmup/5-trial timing QA. No speedup is claimed.','', '## Interpretation','', 'See the accompanying final interpretation below; frozen gate JSON contains all denominators and native-only validity IDs. No quantization, retrieval, fallback, or further training was implemented.']
    (OUT/'stage3_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cells','order','scaling_position']},indent=2))
