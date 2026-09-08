#!/usr/bin/env python
"""Two warmups, five synchronized trials, fixed development item; no speedup claim."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import statistics
import torch
from ccl.stage3_eval import TargetModel,MemoryExtractor,Evaluator,CKPT,OUT,timed,sha
from ccl.stage3_corpus import load_draw
from ccl.provenance import stamp

if __name__=='__main__':
    tm=TargetModel('Qwen/Qwen3-4B')
    ck=torch.load(CKPT,map_location=tm.device,weights_only=False)
    ex=MemoryExtractor(tm,layer_share=ck.get('layer_share',1),n_sink=ck.get('n_sink',8))
    ex.load_state_dict(ck['state_dict']); ex.eval(); ex.requires_grad_(False); del ck
    ev=Evaluator(tm,ex,4)
    item=load_draw(OUT/'stage3_dev_A.json')[0]
    ids=[tm.encode(p) for p in item.pages]
    samples=[]
    with torch.no_grad():
        for i in range(7):
            pg=[]; costs=[]
            for x in ids:
                p,c=timed(lambda:ex.extract(x,4)); pg.append(p); costs.append(c)
            r=ev.ask(item,ids,'INDEPENDENT',pg,sum(costs))
            n=ev.ask(item,ids,'NATIVE')
            if i>=2: samples.append({'page_compile_s':costs,'independent':r['timing'],'native':n['timing']})
    medians={c:{k:statistics.median(s[c][k] for s in samples) for k,v in samples[0][c].items() if isinstance(v,(int,float))} for c in ['independent','native']}
    (OUT/'timing_dev.json').write_text(json.dumps({'samples':samples,'medians':medians,'warmups':2,'trials':5,'item_id':item.item_id,'raw_tokens':sum(x.shape[1] for x in ids),'memory_positions':sum(p[0][0].shape[2] for p in pg),'checkpoint_sha256':sha(CKPT),**stamp()},indent=2)+'\n')
    print(json.dumps(medians,indent=2))
