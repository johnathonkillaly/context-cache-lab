"""Pre-specified Stage 3 aggregation; native-only validity filtering."""
from collections import defaultdict
import math
import statistics


def mean(rows,key='clean_hit'):
    return statistics.mean(float(r[key]) for r in rows) if rows else None


def normalized(q,n,z):
    return (q-z)/(n-z) if all(v is not None for v in (q,n,z)) and n>z else None


def wilson(k,n):
    if not n: return None
    z=1.96; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-half),min(1,mid+half)]


def analyze(rows):
    index=defaultdict(dict)
    for r in rows:
        key=(r['item_id'],r['variant'])
        if r['condition'] in index[key]: raise ValueError('duplicate condition')
        index[key][r['condition']]=r
    primary=[cs for (item,v),cs in index.items() if v=='base' and cs['NATIVE']['task']=='chain' and cs['NATIVE']['hops']==2 and cs['NATIVE']['terminal']=='semantic']
    def valid(cs):
        h=cs['NATIVE']['hops']
        return cs['NATIVE']['clean_hit'] and not cs['NOCTX']['clean_hit'] and all(not cs[f'NATIVE_LOO_{i}']['clean_hit'] for i in range(h))
    valid_primary=[cs for cs in primary if valid(cs)]
    def q(c,subset=valid_primary): return mean([cs[c] for cs in subset])
    nq,zq=q('NATIVE'),q('NOCTX')
    ri=normalized(q('INDEPENDENT'),nq,zq); rj=normalized(q('JOINT'),nq,zq)
    gains=[cs['INDEPENDENT']['rank_margin']-max(cs[f'INDEPENDENT_LOO_{i}']['rank_margin'] for i in range(2)) for cs in valid_primary]
    fraction=sum(g>.1 for g in gains)/len(gains) if gains else None
    scaling=[cs for cs in primary if int(cs['NATIVE']['item_id'].rsplit('_',1)[1])<12]
    q2=q('INDEPENDENT',scaling)
    s16=[index[(cs['NATIVE']['item_id'],'pages_16')] for cs in scaling]
    q16=q('INDEPENDENT',s16)
    retention=q16/q2 if q2 else None
    gaps={c:q('INDEPENDENT')-q(c) if valid_primary else None for c in ['NOCTX','RANDOM','WRONGPAGE']}
    single_rates=[q(f'NATIVE_LOO_{i}',primary) for i in range(2)]
    validity=len(valid_primary)>=8 and all(v is not None and v<.5 for v in single_rates)
    gates={
        '1_retained_quality':{'value':ri,'threshold':.60,'pass':ri is not None and ri>=.60},
        '2_missing_page_dependence':{'value':fraction,'threshold':.80,'pass':fraction is not None and fraction>=.80},
        '3_independent_tax':{'independent_retained':ri,'joint_retained':rj,'threshold':.85,'informative':rj is not None and rj>=.4,'pass':ri is not None and rj is not None and rj>=.4 and ri>=.85*rj},
        '4_scaling':{'q2':q2,'q16':q16,'retention':retention,'threshold':.75,'informative':retention is not None,'pass':retention is not None and retention>=.75},
        '5_controls':{'gaps':gaps,'threshold':.15,'pass':all(v is not None and v>=.15 for v in gaps.values())}}
    cells={}
    for hops in (2,3,4):
        for terminal in ('semantic','identifier','number','hash'):
            subset=[cs for (item,v),cs in index.items() if v=='base' and cs['NATIVE']['task']=='chain' and cs['NATIVE']['hops']==hops and cs['NATIVE']['terminal']==terminal]
            valid_subset=[cs for cs in subset if valid(cs)]
            stats={c:{'q':q(c,subset),'margin':mean([cs[c] for cs in subset],'rank_margin'),
                       'exact_match':mean([cs[c] for cs in subset],'exact_match'),
                       'semantic_correct':mean([cs[c] for cs in subset],'semantic_correct'),
                       'rank_accuracy':mean([cs[c] for cs in subset],'rank_correct')} for c in ['NATIVE','NOCTX','INDEPENDENT','JOINT','BUDGET','RANDOM','WRONGPAGE']}
            stats['n']=len(subset);stats['valid_n']=len(valid_subset)
            stats['retained']=normalized(q('INDEPENDENT',subset),q('NATIVE',subset),q('NOCTX',subset))
            stats['valid_retained']=normalized(q('INDEPENDENT',valid_subset),q('NATIVE',valid_subset),q('NOCTX',valid_subset))
            stats['composition_gain']=q('INDEPENDENT',subset)-max(q(f'INDEPENDENT_LOO_{i}',subset) for i in range(hops)) if subset else None
            stats['native_composition_gain']=q('NATIVE',subset)-max(q(f'NATIVE_LOO_{i}',subset) for i in range(hops)) if subset else None
            stats['mean_item_rank_gain']=statistics.mean(cs['INDEPENDENT']['rank_margin']-max(cs[f'INDEPENDENT_LOO_{i}']['rank_margin'] for i in range(hops)) for cs in subset) if subset else None
            cells[f'{hops}_{terminal}']=stats
    order={}
    order_base=[cs for (item,v),cs in index.items() if v=='base' and cs['NATIVE']['task']=='order']
    for c in ['NATIVE','NOCTX','INDEPENDENT','JOINT','BUDGET','RANDOM','WRONGPAGE']:
        counter=[index[(cs['NATIVE']['item_id'],'shuffled')] for cs in order_base]
        pairs=[bool(cs[c]['clean_hit'] and sh[c]['clean_hit']) for cs,sh in zip(order_base,counter)]
        order[c]={'ordered':q(c,order_base),'shuffled_relabelled':q(c,counter),'paired':statistics.mean(pairs) if pairs else None,'n':len(pairs)}
    scaling_stats={}
    for label in ['base','pages_4','pages_8','pages_16','pages_32','first_middle','middle_middle','middle_last','last_two']:
        sub=[index[(cs['NATIVE']['item_id'],label)] for cs in scaling]
        scaling_stats[label]={c:{'q':q(c,sub),'margin':mean([cs[c] for cs in sub],'rank_margin')} for c in ['NATIVE','INDEPENDENT','JOINT','BUDGET','NOCTX','RANDOM','WRONGPAGE']}
        scaling_stats[label]['n']=len(sub)
    diagnostic=[]
    for h in (3,4):
        cell=cells[f'{h}_semantic']; base=cells['2_semantic']['retained']
        if cell['NATIVE']['q'] is not None and cell['NATIVE']['q']>=.5 and base is not None and cell['retained'] is not None and cell['retained']<.5*base: diagnostic.append(f'{h}-hop semantic collapse')
    if order['NATIVE']['paired'] is not None and order['NATIVE']['paired']>=.5 and order['INDEPENDENT']['paired']<.5*order['NATIVE']['paired']: diagnostic.append('order collapse')
    proof=all(gates[g]['pass'] for g in ['1_retained_quality','2_missing_page_dependence','5_controls'])
    verdict='NOT ESTABLISHED' if not validity else 'PASS' if all(g['pass'] for g in gates.values()) and not diagnostic else 'PARTIAL' if proof else 'FAIL'
    return {'verdict':verdict,'stage2b_verdict':'INCONCLUSIVE','validity':{'sufficient':validity,'primary_n':len(primary),'valid_n':len(valid_primary),'native_single_page_rates':single_rates,'valid_ids':[cs['NATIVE']['item_id'] for cs in valid_primary]},
            'gates':gates,'diagnostic_collapses':diagnostic,'primary_valid_q':{c:q(c) for c in ['NATIVE','NOCTX','INDEPENDENT','JOINT','BUDGET','RANDOM','WRONGPAGE','INDEPENDENT_LOO_0','INDEPENDENT_LOO_1']},
            'primary_valid_wilson95':wilson(sum(cs['INDEPENDENT']['clean_hit'] for cs in valid_primary),len(valid_primary)),
            'missing_page_dependence_wilson95':wilson(sum(g>.1 for g in gains),len(gains)),
            'cells':cells,'order':order,'scaling_position':scaling_stats}
