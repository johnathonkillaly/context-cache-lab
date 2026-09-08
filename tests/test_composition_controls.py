from ccl.stage3_eval import clean_prediction,condition_specs
from ccl.stage3_corpus import generate


def test_all_chain_ablation_controls_and_no_vacuous_shuffle():
    for item in generate('stage3_dev_A'):
        cs=condition_specs(item)
        assert {'NATIVE','NOCTX','INDEPENDENT','JOINT','BUDGET','RANDOM','WRONGPAGE'}<=set(cs)
        if item.task=='chain':
            assert all(f'{c}_LOO_{i}' in cs for i in range(item.hops) for c in ('NATIVE','INDEPENDENT'))
        assert 'SHUFFLE_PAGES' not in cs


def test_exact_boundaries_and_competing_options():
    assert clean_prediction('Value: 1234.','1234',['1234','5678'],'number')
    assert not clean_prediction('12345','1234',['1234','5678'],'number')
    assert not clean_prediction('1234 or 5678','1234',['1234','5678'],'number')
    assert clean_prediction('The Steel Cooling Pump.','steel cooling pump',['steel cooling pump','linen brush'],'semantic')
