from ccl.stage3_report import normalized,wilson
from ccl.stage3_eval import clean_prediction


def test_retention_is_undefined_at_no_native_gain_not_a_pass():
    assert normalized(.5,.2,.2) is None
    assert normalized(.1,.8,.2)<0
    assert normalized(.9,.8,.2)>1
    assert wilson(0,0) is None
    assert wilson(0,12)[1]>.2


def complete_rows(independent=True,joint=True,native=True):
    from ccl.stage3_corpus import generate,variants
    from ccl.stage3_eval import condition_specs
    rows=[]
    for item in generate('stage3_dev_A'):
        for label,v in variants(item):
            for c in condition_specs(v):
                hit=(native if c=='NATIVE' else independent if c=='INDEPENDENT' else joint if c=='JOINT' else False)
                rows.append({'item_id':item.item_id,'variant':label,'condition':c,'task':item.task,'hops':item.hops,
                             'terminal':item.terminal,'clean_hit':hit,'rank_margin':2 if hit else -1,
                             'exact_match':hit,'semantic_correct':hit,'rank_correct':hit})
    return rows


def test_weak_joint_is_noninformative_not_manufactured_pass():
    from ccl.stage3_report import analyze
    result=analyze(complete_rows(joint=False))
    assert result['verdict']=='PARTIAL'
    assert result['gates']['3_independent_tax']['informative'] is False
    assert result['gates']['3_independent_tax']['pass'] is False


def test_zero_compiled_quality_cannot_pass_scaling_or_composition():
    from ccl.stage3_report import analyze
    result=analyze(complete_rows(independent=False,joint=False))
    assert result['verdict']=='FAIL'
    assert result['gates']['4_scaling']['retention'] is None
    assert result['gates']['4_scaling']['pass'] is False


def test_native_failure_does_not_become_carrier_failure():
    from ccl.stage3_report import analyze
    result=analyze(complete_rows(native=False,independent=False))
    assert result['verdict']=='NOT ESTABLISHED'
    assert result['validity']['valid_n']==0


def test_native_only_validity_is_independent_of_compiled_scores():
    from ccl.stage3_report import analyze
    a=analyze(complete_rows(independent=False))
    b=analyze(complete_rows(independent=True))
    assert a['validity']['valid_ids']==b['validity']['valid_ids']
