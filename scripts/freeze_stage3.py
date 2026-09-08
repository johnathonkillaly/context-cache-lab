#!/usr/bin/env python
"""Freeze new draws and existing checkpoint, without reading historic Draw B."""
import json
from pathlib import Path
import torch
from transformers import AutoConfig
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]/'src'))
from ccl.stage3_corpus import write_draw
from ccl.stage3_eval import OUT,CKPT,sha
from ccl.provenance import stamp

if __name__=='__main__':
    path=OUT/'freeze.json'
    if path.exists(): raise SystemExit('Already frozen; refusing overwrite')
    corpora={d+'.json':write_draw(d,OUT/(d+'.json')) for d in ('stage3_dev_A','stage3_test_B')}
    ck=torch.load(CKPT,map_location='cpu',weights_only=False)
    cfg=AutoConfig.from_pretrained('Qwen/Qwen3-4B',local_files_only=True)
    record={'corpora':corpora,'checkpoint_sha256':sha(CKPT),'checkpoint_step':ck['step'],
            'model_revision':cfg._commit_hash,'checkpoint_path':str(CKPT),
            'protocol_sha256':sha(Path('docs/STAGE3_PROTOCOL.md')),
            'historical_files':{str(p):sha(p) for p in sorted(Path('results/raw').glob('stage*.json'))},**stamp()}
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('historical_files','env')},indent=2))
