from dataclasses import asdict
from ccl.stage3_corpus import generate, reachable, variants


def test_determinism_and_symbolic_necessity():
    a=generate('stage3_dev_A')
    assert [asdict(x) for x in a]==[asdict(x) for x in generate('stage3_dev_A')]
    for item in a:
        if item.task!='chain': continue
        assert reachable(item)=={item.answer}
        assert len(set(item.options))==4
        assert item.answer not in item.question
        for omitted in range(item.hops):
            assert len(reachable(item,[omitted]))==4
        # All terminal candidates share one page; that page alone cannot select gold.
        assert all(v in item.pages[-1] for v in item.options)
        assert all(item.answer not in p for p in item.pages[:-1])


def test_separate_draw_values():
    a,b=[generate(d) for d in ('stage3_dev_A','stage3_test_B')]
    for terminal in ('semantic','identifier','number','hash'):
        va={v for x in a if x.task=='chain' and x.terminal==terminal for v in x.options}
        vb={v for x in b if x.task=='chain' and x.terminal==terminal for v in x.options}
        assert not va & vb


def test_scaling_keeps_information_and_position_variants():
    item=generate('stage3_dev_A')[0]
    vs=dict(variants(item))
    for name,v in vs.items():
        assert [v.pages[i] for i in v.relevant]==item.pages
        assert v.answer==item.answer
    assert [len(vs[f'pages_{n}'].pages) for n in (4,8,16,32)]==[4,8,16,32]
    assert vs['last_two'].relevant==[14,15]


def test_order_counterfactual_changes_gold_not_query_or_pages():
    item=generate('stage3_dev_A')[-1]
    vs=dict(variants(item))
    other=vs['shuffled']
    assert other.answer!=item.answer and other.question==item.question
    assert sorted(other.pages)==sorted(item.pages)
    assert other.answer in other.pages[2]
    assert not any('then' in p or 'PAGE ' in p for p in item.pages)
