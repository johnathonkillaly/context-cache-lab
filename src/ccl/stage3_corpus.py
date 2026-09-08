"""Deterministic, branching cross-page tasks. No dependency on Stage 1/2 draws."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import random
from pathlib import Path

DRAWS = {'stage3_dev_A': 310701, 'stage3_test_B': 310702}
TEMPLATES = [
    ('project', 'engineer', '{a} is assigned to {b}.', 'the engineer assigned to {a}'),
    ('device', 'technician', '{b} maintains {a}.', 'the technician maintaining {a}'),
    ('city', 'facility', '{a} uses {b}.', 'the facility used by {a}'),
    ('component', 'supplier', '{b} provides {a}.', 'the supplier providing {a}'),
    ('animal', 'owner', '{a} belongs to {b}.', 'the owner of {a}'),
    ('visitor', 'office', '{a} is registered with {b}.', 'the office registering {a}'),
]
FILLER = ('The local register is maintained on paper. Staff check spelling during routine reviews. '
          'These records describe fictional assignments. The notice board has a plain border. '
          'Unused stationery is kept in a separate cupboard. The room is cleaned regularly.')

@dataclass
class Item:
    item_id: str
    draw: str
    seed: int
    task: str
    terminal: str
    hops: int
    template: int
    pages: list[str]
    relevant: list[int]
    question: str
    answer: str
    options: list[str]
    # Explicit graph retained for symbolic necessity tests, never sent to the model.
    edges: list[list[list[str]]]
    start: str
    distractors: list[str]

    @property
    def text(self):
        return ''.join(self.pages)


def page(sentences):
    return '\n'.join(sentences) + '\n' + FILLER + '\n\n'


def make_item(draw, index, hops=2, terminal='semantic'):
    seed = DRAWS[draw] + index * 101 + hops * 10007 + ['semantic','identifier','number','hash'].index(terminal)*100003
    rng = random.Random(seed)
    namespace = 'V' if draw == 'stage3_dev_A' else 'Z'
    used = set()
    def name(kind):
        while True:
            s = namespace + ''.join(rng.choice(['al','en','ir','os','um','ev','ak','il','or','un']) for _ in range(3))
            if s not in used:
                used.add(s)
                return kind + ' ' + s
    ti = index % len(TEMPLATES)
    root, bridge, sentence, query = TEMPLATES[ti]
    levels = [[name(root) for _ in range(4)], [name(bridge) for _ in range(4)]]
    for kind in ['depot', 'cabinet'][:hops-2]:
        levels.append([name(kind) for _ in range(4)])
    if terminal == 'semantic':
        materials = ['ceramic','copper','linen','rubber','wooden','glass'] if namespace=='V' else ['steel','bronze','cotton','silicone','bamboo','porcelain']
        vals = [f'{a} {b} {c}' for a in materials for b in ['cooling','inspection','storage','cleaning','measuring','repair'] for c in ['pump','tray','brush','box','tool','panel']]
        values = rng.sample(vals,4)
    elif terminal == 'identifier':
        values = [f'{namespace}{rng.randrange(100,999)}-{rng.randrange(100,999)}-K' for _ in range(4)]
    elif terminal == 'number':
        lo = 100000 if namespace=='V' else 600000
        values = [str(v) for v in rng.sample(range(lo,lo+300000),4)]
    else:
        values = [namespace.lower().replace('v','a').replace('z','f') + f'{rng.getrandbits(60):015x}' for _ in range(4)]
    levels.append(values)
    edge_pages, pages = [], []
    cur = rng.choice(levels[0]); start = cur
    label = {'semantic':'delivery item','identifier':'delivery identifier','number':'delivery quantity','hash':'delivery hash'}[terminal]
    for h in range(hops):
        dest = levels[h+1].copy(); rng.shuffle(dest)
        pairs = list(zip(levels[h],dest)); edge_pages.append([list(p) for p in pairs])
        cur = dict(pairs)[cur]
        rng.shuffle(pairs)
        if h == 0:
            lines = [sentence.format(a=a,b=b) for a,b in pairs]
        elif h == hops-1:
            lines = [f'The {label} assigned to {a} is {b}.' for a,b in pairs]
        else:
            lines = [f'{a} uses {b}.' for a,b in pairs]
        pages.append(page(lines))
    target = query.format(a=start)
    if hops>=3: target = 'the depot used by ' + target
    if hops>=4: target = 'the cabinet used by ' + target
    question = f'What is the {label} assigned to {target}?'
    # Same-format distractors; disjoint entities/values, fixed pool reused as N increases.
    distractors = []
    for j in range(30):
        lines = [f'{name(root)} is assigned to {name(bridge)}.' for _ in range(4)]
        distractors.append(page(lines))
    return Item(f'{draw}_{terminal}_{hops}_{index:03}',draw,seed,'chain',terminal,hops,ti,pages,list(range(hops)),question,cur,values,edge_pages,start,distractors)


def make_order(draw,index):
    rng = random.Random(DRAWS[draw]+900000+index)
    verbs = ['activates','inspects','closes','cleans','opens','checks','replaces','disconnects']
    objects = ['cooling loop','pressure valve','backup pump','intake filter','service hatch','warning lamp','drain pipe','control panel']
    events = rng.sample([f'The operator {v} the {o}' for v in verbs for o in objects],4)
    # No numbering, timestamps, or links in any page. Query explicitly defines document-order semantics.
    pages = [page([e+'.']) for e in events]
    return Item(f'{draw}_order_{index:03}',draw,DRAWS[draw]+900000+index,'order','semantic',0,index%6,pages,[1,2],
                f'In document page order, what event occurs immediately after "{events[1]}"? Return the event sentence.',events[2],events,[],events[1],[])


def generate(draw):
    items = [make_item(draw,i) for i in range(24)]
    for hops in (2,3,4):
        for terminal in ('semantic','identifier','number','hash'):
            if hops==2 and terminal=='semantic': continue
            items.extend(make_item(draw,i,hops,terminal) for i in range(6))
    items.extend(make_order(draw,i) for i in range(12))
    return items


def variants(item, suite='all'):
    if suite in ('all','composition'):
        yield 'base',item
        if item.task=='order':
            yield 'shuffled',replace(item,pages=[item.pages[i] for i in [0,1,3,2]],answer=item.options[3])
    if suite in ('all','scaling') and item.task=='chain' and item.hops==2 and item.terminal=='semantic' and int(item.item_id.rsplit('_',1)[1])<12:
        for n in (4,8,16,32):
            yield f'pages_{n}', replace(item,pages=item.pages+item.distractors[:n-2])
        n=16
        for label,slots in [('first_middle',[0,8]),('middle_middle',[7,8]),('middle_last',[7,15]),('last_two',[14,15])]:
            filler=iter(item.distractors[:14]); rel=dict(zip(slots,item.pages))
            yield label,replace(item,pages=[rel[i] if i in rel else next(filler) for i in range(n)],relevant=slots)


def reachable(item, omitted=()):
    possible={item.start}
    for i,edges in enumerate(item.edges):
        possible = {b for a,b in edges if a in possible or i in omitted}
    return possible


def write_draw(draw,path):
    data = json.dumps([asdict(i) for i in generate(draw)],indent=1)+'\n'
    Path(path).write_text(data)
    return hashlib.sha256(data.encode()).hexdigest()


def load_draw(path):
    return [Item(**d) for d in json.loads(Path(path).read_text())]
