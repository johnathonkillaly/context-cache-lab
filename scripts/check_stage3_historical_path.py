#!/usr/bin/env python
"""Post-run development-only numerical equivalence to historical Stage 2b query path.
No held-out examples and no training; does not amend any quality score or gate.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import json
import torch
from ccl.stage3_eval import TargetModel,MemoryExtractor,Evaluator,CKPT,OUT,sha
from ccl.stage3_corpus import load_draw
from ccl.corpus import Fact
from ccl.provenance import stamp
from run_stage2b_eval import _head_layers,_ask_composed

if __name__=='__main__':
    tm=TargetModel('Qwen/Qwen3-4B')
    ck=torch.load(CKPT,map_location=tm.device,weights_only=False)
    ex=MemoryExtractor(tm,layer_share=ck.get('layer_share',1),n_sink=ck.get('n_sink',8))
    ex.load_state_dict(ck['state_dict']); ex.eval(); ex.requires_grad_(False); del ck
    ev=Evaluator(tm,ex,4)
    item=load_draw(OUT/'stage3_dev_A.json')[0]
    ids=[tm.encode(p) for p in item.pages]
    with torch.no_grad():
        compiled=[ev.compile(x) for x in ids]; pages=[p for p,_ in compiled]
        new=ev.ask(item,ids,'INDEPENDENT',pages,sum(c for _,c in compiled))
        worst=max((v for v in item.options if v!=item.answer),key=new['option_logprobs'].get)
        hint='the delivery item phrase'
        f=Fact(item.item_id,'semantic',item.question,item.answer,hint,worst,item.relevant,[])
        head,_,tail,suffix=tm.split_prompt4(item.question,item.text,hint)
        hl,hn=_head_layers(tm,head)
        old=_ask_composed(tm,hl,hn,pages,tail,suffix,f,48)
    assert new['prediction']==old['prediction']
    assert abs(new['rank_margin']-old['rank_margin'])<1e-6
    assert abs(new['option_logprobs'][item.answer]-old['answer_logprob'])<1e-6
    record={'scope':'post-run development-only equivalence, not a new held-out evaluation',
            'item_id':item.item_id,'new_prediction':new['prediction'],'historical_prediction':old['prediction'],
            'new_rank_margin':new['rank_margin'],'historical_rank_margin':old['rank_margin'],
            'gold_logprob_delta':new['option_logprobs'][item.answer]-old['answer_logprob'],
            'passed':True,'checkpoint_sha256':sha(CKPT),**stamp()}
    path=OUT/'historical_path_qa.json'
    with path.open('x') as out: json.dump(record,out,indent=2)
    print(json.dumps(record,indent=2))
