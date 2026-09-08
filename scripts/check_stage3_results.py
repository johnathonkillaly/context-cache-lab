#!/usr/bin/env python
"""Audit saved Stage 3 rows without any model evaluation or artifact rewriting."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import math
from collections import defaultdict
from ccl.stage3_eval import OUT,CKPT,sha,source_hashes,condition_specs
from ccl.stage3_corpus import load_draw,variants

if __name__=='__main__':
    f=json.loads((OUT/'freeze.json').read_text())
    assert sha(CKPT)==f['checkpoint_sha256']
    assert source_hashes()==f['evaluation_sources']
    for p,digest in f['historical_files'].items(): assert sha(p)==digest,p
    for p,digest in f['corpora'].items(): assert sha(OUT/p)==digest,p
    path=OUT/'stage3_test_B_r4.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    index=defaultdict(dict)
    for r in rows:
        k=(r['item_id'],r['variant'])
        assert r['condition'] not in index[k]
        index[k][r['condition']]=r
        assert math.isfinite(r['rank_margin'])
        assert r['rank_margin']==r['option_logprobs'][r['gold']]-max(v for k,v in r['option_logprobs'].items() if k!=r['gold'])
        assert r['active_prompt_positions']>r['compiled_state_positions']
        assert r['total_pages']==r['relevant_pages']+r['distractor_pages']
        assert r['checkpoint_sha256']==f['checkpoint_sha256']
        if r['compiled_state_positions']:
            assert r['timing']['target_original_page_forward_tokens']==0
        if r['condition'] in ('RANDOM','WRONGPAGE'): assert r['supplied_relevant_pages']==0
    for k,cs in index.items():
        if 'INDEPENDENT' not in cs: continue
        n=cs['INDEPENDENT']['compiled_state_positions']
        if 'BUDGET' in cs:
            assert cs['BUDGET']['raw_context_tokens_supplied']==n
            assert cs['BUDGET']['active_prompt_positions']==cs['INDEPENDENT']['active_prompt_positions']
        for c in ('RANDOM','WRONGPAGE'):
            if c in cs: assert cs[c]['compiled_state_positions']==n
    complete=path.with_suffix('.complete.json')
    if complete.exists():
        assert sha(path)==json.loads(complete.read_text())['sha256']
        expected={}
        for it in load_draw(OUT/'stage3_test_B.json'):
            for label,v in variants(it): expected[(it.item_id,label)]=set(condition_specs(v))
        assert set(index)==set(expected)
        assert all(set(index[k])==v for k,v in expected.items())
        print(f'COMPLETE AUDIT: {len(rows)} rows, {len(index)} variants, budgets/controls/coverage/historical hashes valid')
    else:
        print(f'PARTIAL AUDIT: {len(rows)} existing rows; budgets, controls, source and historical hashes valid')
