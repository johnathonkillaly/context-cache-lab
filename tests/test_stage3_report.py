from ccl.stage3_report import normalized,wilson
from ccl.stage3_eval import clean_prediction


def test_retention_is_undefined_at_no_native_gain_not_a_pass():
    assert normalized(.5,.2,.2) is None
    assert normalized(.1,.8,.2)<0
    assert normalized(.9,.8,.2)>1
    assert wilson(0,0) is None
    assert wilson(0,12)[1]>.2
