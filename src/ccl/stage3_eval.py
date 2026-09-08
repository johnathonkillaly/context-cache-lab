"""Frozen Stage 3 evaluator; append-only resumable measurements."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
from .compressor import MemoryExtractor
from .metrics import norm_exact, norm_semantic, _contains_semantic
from .provenance import stamp
from .rope import apply_rope_to_keys, cos_sin_for, cache_from_layers
from .stage3_corpus import load_draw, variants
from .target import TargetModel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/stage3'
CKPT = ROOT/'results/raw/stage2b_extractor.pt'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def source_hashes():
    paths = [*sorted((ROOT/'src/ccl').glob('*.py')),ROOT/'docs/STAGE3_PROTOCOL.md',ROOT/'scripts/stage3_composition_eval.py',ROOT/'scripts/stage3_page_scaling.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def sync():
    if torch.backends.mps.is_available(): torch.mps.synchronize()


def timed(fn):
    sync(); start=time.perf_counter(); result=fn(); sync()
    return result,time.perf_counter()-start


def compose_timed(model,pages,start):
    """Same operations as compose_pages, with rotation separated from concatenation."""
    def rotate():
        pos=start; positioned=[]
        for page in pages:
            n=page[0][0].shape[2]
            cos,sin=cos_sin_for(model,torch.arange(pos,pos+n,device=model.device),page[0][0].dtype)
            positioned.append([(apply_rope_to_keys(k,cos,sin),v) for k,v in page]); pos+=n
        return positioned,pos
    (positioned,nxt),rot_s=timed(rotate)
    def merge():
        return [(torch.cat([p[i][0] for p in positioned],2),torch.cat([p[i][1] for p in positioned],2)) for i in range(len(pages[0]))]
    layers,cat_s=timed(merge)
    return layers,nxt,{'rope_s':rot_s,'concat_s':cat_s}


def random_pages(pages,seed):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    out=[]
    for page in pages:
        layers=[]
        for pair in page:
            values=[]
            for t in pair:
                mean=t.float().mean(2,keepdim=True)
                std=t.float().std(2,keepdim=True,unbiased=False).clamp_min(1e-6)
                noise=torch.randn(t.shape,generator=generator).to(t.device)
                values.append((mean+std*noise).to(t.dtype))
            layers.append(tuple(values))
        out.append(layers)
    return out


def clean_prediction(pred,answer,options,terminal):
    def contains(v):
        if terminal=='semantic': return _contains_semantic(pred,v)
        # Strict token boundary avoids partial numeric or identifier matches.
        import re
        return re.search(r'(?<![\w-])'+re.escape(norm_exact(v))+r'(?![\w-])',norm_exact(pred)) is not None
    return contains(answer) and not any(contains(v) for v in options if v!=answer)


class Evaluator:
    def __init__(self,tm,ex,ratio=4):
        self.tm,self.ex,self.ratio=tm,ex,ratio
        self.page_cache={}; self.compile_times={}

    @torch.no_grad()
    def compile(self,ids):
        key=tuple(ids[0].tolist())
        if key not in self.page_cache:
            page,seconds=timed(lambda:self.ex.extract(ids,self.ratio))
            self.page_cache[key]=page; self.compile_times[key]=seconds
        return self.page_cache[key],self.compile_times[key]

    @torch.no_grad()
    def ask(self,item,page_ids,condition,pages=None,compile_s=0):
        tm=self.tm
        hint={'semantic':'the delivery item phrase','identifier':'the delivery identifier','number':'the delivery quantity','hash':'the delivery hash'}[item.terminal]
        if item.task=='order': hint='the event sentence'
        # Stable protocol scaffolding; every context uses precisely the same token head/tail.
        parts=tm.split_prompt4(item.question,item.text,hint)
        if parts is None: raise RuntimeError('non-tokenizable prompt boundary')
        head,_,tail,suffix=parts
        hids=tm.encode(head); feed=torch.cat([tm.encode(tail),tm.encode(suffix)],1)
        times={'compile_s':compile_s,'serialized_load_s':None,'rope_s':0.,'concat_s':0.,'cache_setup_s':0.,'head_prefill_s':0.,'target_original_page_forward_tokens':0}
        if condition=='NOCTX':
            ids=tm.encode(tm.build_prompt(item.question,None,hint))
            (logits,cache),times['target_tail_s']=timed(lambda:tm.prefill(ids))
            context_n=0; positions=0
        elif pages is None:
            body=torch.cat(page_ids,1) if page_ids else hids[:,:0]
            ids=torch.cat([hids,body,feed],1)
            (logits,cache),times['target_tail_s']=timed(lambda:tm.prefill(ids))
            context_n=body.shape[1]; positions=0
            times['target_original_page_forward_tokens']=context_n
        else:
            (_,head_cache),times['head_prefill_s']=timed(lambda:tm.prefill(hids))
            hlayers=[(x.keys,x.values) for x in head_cache.layers]
            layers,nxt,ct=compose_timed(tm.model,pages,hids.shape[1]); times.update(ct)
            def setup():
                merged=[(torch.cat([h[0],p[0]],2),torch.cat([h[1],p[1]],2)) for h,p in zip(hlayers,layers)]
                return cache_from_layers(tm.model,merged)
            cache,times['cache_setup_s']=timed(setup)
            pos=torch.arange(nxt,nxt+feed.shape[1],device=tm.device)
            (logits,cache),times['target_tail_s']=timed(lambda:tm.prefill(feed,cache,position_ids=pos,cache_position=pos))
            positions=sum(p[0][0].shape[2] for p in pages); context_n=0
        _,times['first_token_s']=timed(lambda:int(logits.argmax(-1)[0]))
        active=int(cache.get_seq_length()) # BEFORE scoring or decode mutates the cache
        times['warm_ttft_s']=sum(times[k] for k in ['head_prefill_s','rope_s','concat_s','cache_setup_s','target_tail_s','first_token_s'])
        times['cold_ttft_s']=times['warm_ttft_s']+compile_s
        logprobs={v:tm.answer_logprob(v,prompt_last_logits=logits,cache=cache)['mean_logprob'] for v in item.options}
        margin=logprobs[item.answer]-max(x for v,x in logprobs.items() if v!=item.answer)
        prediction=tm.decode_from(logits,cache,max_new_tokens=48)
        result={'prediction':prediction['text'],'n_generated':prediction['n_generated'],'gold':item.answer,
                'option_logprobs':logprobs,'rank_margin':margin,'rank_correct':margin>0,
                'exact_match':norm_exact(prediction['text'])==norm_exact(item.answer),
                'semantic_correct':_contains_semantic(prediction['text'],item.answer) and not any(_contains_semantic(prediction['text'],v) for v in item.options if v!=item.answer),
                'clean_hit':clean_prediction(prediction['text'],item.answer,item.options,item.terminal),
                'raw_context_tokens_supplied':int(context_n),'compiled_state_positions':int(positions),
                'active_prompt_positions':active,'timing':times}
        return result


def condition_specs(item):
    base=['NATIVE','NOCTX','INDEPENDENT','JOINT','BUDGET','RANDOM','WRONGPAGE']
    if item.task=='chain':
        base += [f'{c}_LOO_{i}' for i in range(item.hops) for c in ('NATIVE','INDEPENDENT')]
    return base


@torch.no_grad()
def evaluate_variant(ev,item,other,label,emit,done,native_only=False):
    tm=ev.tm
    ids=[tm.encode(p) for p in item.pages]
    raw_n=sum(x.shape[1] for x in ids)
    pages=None; costs=None
    def learned():
        nonlocal pages,costs
        if pages is None:
            compiled=[ev.compile(x) for x in ids]; pages=[p for p,_ in compiled]; costs=[s for _,s in compiled]
        return pages,costs
    for condition in condition_specs(item):
        if native_only and not(condition.startswith('NATIVE') or condition=='NOCTX'): continue
        key=(item.item_id,label,condition)
        if key in done: continue
        supplied=list(range(len(ids))); condition_pages=None; cs=0; page_costs=[]
        if '_LOO_' in condition:
            removed=item.relevant[int(condition.rsplit('_',1)[1])]; supplied.remove(removed)
        use_ids=[ids[i] for i in supplied]
        if condition.startswith('INDEPENDENT') or condition in ('RANDOM','WRONGPAGE','BUDGET'):
            pg,co=learned()
            condition_pages=[pg[i] for i in supplied]; page_costs=[co[i] for i in supplied]; cs=sum(page_costs)
        if condition=='JOINT':
            jp,jc=ev.compile(torch.cat(ids,1)); condition_pages=[jp]; cs=jc; page_costs=[jc]
        elif condition=='RANDOM':
            condition_pages=random_pages(condition_pages,item.seed)
            page_costs=[]; cs=0 # random synthesis isn't learned compilation, separately not a speed comparator
        elif condition=='WRONGPAGE':
            wp=[]; wc=[]
            for i,x in enumerate(ids):
                source=tm.encode(other.pages[i%len(other.pages)])
                # Match the exact raw-token length, hence memory slots; pad only neutral text.
                pad=tm.encode(' The room is quiet.\n')
                while source.shape[1]<x.shape[1]: source=torch.cat([source,pad],1)
                p,c=ev.compile(source[:,:x.shape[1]]); wp.append(p); wc.append(c)
            condition_pages=wp; page_costs=wc; cs=sum(wc)
        elif condition=='BUDGET':
            n=sum(p[0][0].shape[2] for p in condition_pages)
            use_ids=[torch.cat(ids,1)[:,:n]]; condition_pages=None; cs=0; page_costs=[]
        elif condition=='NOCTX':
            supplied=[]; use_ids=[]
        row=ev.ask(item,use_ids,condition,condition_pages,cs)
        if condition=='BUDGET':
            # Count original pages wholly/partly retained by the prefix.
            remain=row['raw_context_tokens_supplied']; supplied=[]
            for i,x in enumerate(ids):
                if remain<=0: break
                supplied.append(i); remain-=x.shape[1]
        row['timing']['page_compile_s']=page_costs
        rel=0 if condition in ('RANDOM','WRONGPAGE') else sum(i in supplied for i in item.relevant)
        offsets=[]; offset=0
        if condition_pages is not None:
            for p in condition_pages: offsets.append(offset); offset+=p[0][0].shape[2]
        row.update(item_id=item.item_id,variant=label,condition=condition,draw=item.draw,seed=item.seed,
                   task=item.task,terminal=item.terminal,hops=item.hops,template=item.template,
                   ratio=ev.ratio,raw_context_tokens_available=raw_n,
                   realized_context_ratio=raw_n/(row['compiled_state_positions'] or row['raw_context_tokens_supplied']) if row['compiled_state_positions'] or row['raw_context_tokens_supplied'] else None,
                   total_pages=len(item.pages),relevant_pages=len(item.relevant),distractor_pages=len(item.pages)-len(item.relevant),
                   supplied_pages=len(supplied),supplied_relevant_pages=rel,supplied_distractor_pages=len(supplied)-rel,
                   relevant_page_indices=item.relevant,composed_page_offsets=offsets,
                   supplied_page_indices=supplied,wrongpage_source=other.item_id if condition=='WRONGPAGE' else None)
        emit(row)


def main(default_suite='all'):
    ap=argparse.ArgumentParser()
    ap.add_argument('--draw',choices=['stage3_dev_A','stage3_test_B'],default='stage3_dev_A')
    ap.add_argument('--suite',choices=['all','composition','scaling'],default=default_suite)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--ratio',type=int,choices=[2,4,8,16],default=4)
    ap.add_argument('--native-only',action='store_true')
    ap.add_argument('--final',action='store_true')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--out')
    args=ap.parse_args()
    if args.ratio!=4 and not (OUT/'stage3_report.md').exists(): ap.error('secondary ratios require the primary report')
    frozen=json.loads((OUT/'freeze.json').read_text())
    path=OUT/(args.draw+'.json')
    if sha(path)!=frozen['corpora'][path.name] or sha(CKPT)!=frozen['checkpoint_sha256']:
        ap.error('frozen corpus/checkpoint mismatch')
    if args.draw=='stage3_test_B':
        if frozen.get('evaluation_sources')!=source_hashes() or frozen['protocol_sha256']!=sha(ROOT/'docs/STAGE3_PROTOCOL.md'): ap.error('final evaluation sources/protocol differ from freeze')
        if not args.final or args.limit or args.native_only or args.suite!='all' or args.out: ap.error('held-out requires complete --final run')
    out=Path(args.out) if args.out else OUT/(args.draw+f'_r{args.ratio}.jsonl')
    marker=out.with_suffix('.run.json')
    config={'draw':args.draw,'ratio':args.ratio,'suite':args.suite,'limit':args.limit,'native_only':args.native_only,
            'sources':source_hashes(),'checkpoint_sha256':frozen['checkpoint_sha256'],'corpus_sha256':sha(path)}
    if marker.exists():
        if not args.resume or json.loads(marker.read_text())['config']!=config: ap.error('existing run; resume requires exact frozen inputs')
    else:
        if out.exists() or args.resume: ap.error('invalid output/resume state')
        with marker.open('x') as f: json.dump({'config':config,**stamp()},f,indent=2)
    done=set()
    if out.exists():
        for line in out.read_text().splitlines():
            r=json.loads(line); key=(r['item_id'],r['variant'],r['condition'])
            if key in done: raise ValueError('duplicate result')
            done.add(key)
    torch.manual_seed(0)
    print('Loading frozen target and extractor',flush=True)
    tm=TargetModel('Qwen/Qwen3-4B')
    revision=getattr(tm.model.config,'_commit_hash',None)
    if revision!=frozen['model_revision']: raise ValueError('target model revision changed')
    ck=torch.load(CKPT,map_location=tm.device,weights_only=False)
    ex=MemoryExtractor(tm,layer_share=ck.get('layer_share',1),n_sink=ck.get('n_sink',8))
    ex.load_state_dict(ck['state_dict']); ex.eval(); ex.requires_grad_(False)
    assert ck['step']==frozen['checkpoint_step']
    del ck
    ev=Evaluator(tm,ex,args.ratio)
    items=load_draw(path)
    chosen=items[:args.limit] if args.limit else items
    provenance=stamp(model_id=tm.model_id,model_revision=revision,checkpoint_sha256=frozen['checkpoint_sha256'],target_dtype=str(tm.dtype),extractor_dtype=str(ex.param_dtype))
    def emit(row):
        row.update(provenance); row['timestamp_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        with out.open('a') as f:
            f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
    for idx,item in enumerate(chosen):
        others=[x for x in items if x.item_id!=item.item_id and x.task==item.task and x.terminal==item.terminal and x.hops==item.hops and item.answer not in x.text]
        other=others[idx%len(others)]
        for label,v in variants(item,args.suite):
            print(f'[{idx+1}/{len(chosen)}] {item.item_id} {label}',flush=True)
            evaluate_variant(ev,v,other,label,emit,done,args.native_only)
        # Reuse across all variants of one item. Avoid retaining the entire corpus on MPS.
        ev.page_cache.clear(); ev.compile_times.clear(); gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
    expected=sum(sum(1 for c in condition_specs(v) if not args.native_only or c.startswith('NATIVE') or c=='NOCTX') for item in chosen for _,v in variants(item,args.suite))
    count=sum(1 for _ in out.open())
    if count!=expected: raise ValueError(f'incomplete rows {count}/{expected}')
    completion={'rows':count,'sha256':sha(out),'config':config,**stamp()}
    out.with_suffix('.complete.json').write_text(json.dumps(completion,indent=2))
    print(f'COMPLETE {count} rows: {out}',flush=True)
    return 0
