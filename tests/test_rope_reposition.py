from types import SimpleNamespace
import pytest
import torch
from ccl.rope import compose_pages, position_keys
from ccl.stage3_eval import compose_timed


def fake_model():
    def rotary(probe,pos):
        inv=1/(1000000**(torch.arange(0,8,2)/8))
        f=pos.float().unsqueeze(-1)*inv
        a=torch.cat([f,f],-1)
        return a.cos(),a.sin()
    return SimpleNamespace(device=torch.device('cpu'),model=SimpleNamespace(rotary_emb=rotary))


def test_composition_repositions_once_without_mutation():
    model=fake_model(); torch.manual_seed(5)
    a=[(torch.randn(1,2,3,8),torch.randn(1,2,3,8)) for _ in range(2)]
    b=[(torch.randn(1,2,5,8),torch.randn(1,2,5,8)) for _ in range(2)]
    saved=[[(k.clone(),v.clone()) for k,v in p] for p in (a,b)]
    ab,n=compose_pages(model,[a,b],start=7)
    ba,_=compose_pages(model,[b,a],start=7)
    timed,nt,_=compose_timed(model,[a,b],start=7)
    assert n==nt==15
    for i in range(2):
        once=position_keys(model,a[i][0],7)
        assert torch.equal(ab[i][0][:,:,:3],once)
        assert not torch.allclose(ab[i][0][:,:,:3],position_keys(model,once,7))
        assert not torch.allclose(ab[i][0][:,:,:3],ba[i][0][:,:,5:])
        assert torch.equal(ab[i][1][:,:,:3],ba[i][1][:,:,5:])
        assert torch.equal(ab[i][0],timed[i][0])
        for p,copy in zip((a,b),saved):
            assert torch.equal(p[i][0],copy[i][0]) and torch.equal(p[i][1],copy[i][1])


@pytest.mark.slow
def test_learned_extractor_stores_prerope_and_is_independent():
    from ccl.target import TargetModel
    from ccl.compressor import MemoryExtractor
    from ccl.stage3_eval import CKPT
    from ccl.rope import PreRopeCapture
    tm=TargetModel('Qwen/Qwen3-4B')
    ck=torch.load(CKPT,map_location=tm.device,weights_only=False)
    ex=MemoryExtractor(tm,layer_share=ck.get('layer_share',1),n_sink=ck.get('n_sink',8))
    ex.load_state_dict(ck['state_dict']); ex.eval(); ex.requires_grad_(False)
    ids=tm.encode('Project Velin is assigned to engineer Zalem.\n')
    with torch.no_grad(),PreRopeCapture(tm.model) as capture:
        a=ex.extract(ids,4)
        captured=capture.stacked()
    for (k,v),(pre,_) in zip(a,captured):
        assert torch.equal(k,pre) # hook's last k_norm call is extractor memory, before rotation
    saved=[(k.clone(),v.clone()) for k,v in a]
    with torch.no_grad():
        ex.extract(tm.encode('Unrelated page about a green cabinet.\n'),4)
        again=ex.extract(ids,4)
    compose_pages(tm.model,[a,again],17)
    for (k,v),(k0,v0),(k1,v1) in zip(a,saved,again):
        assert torch.equal(k,k0) and torch.equal(v,v0)
        assert torch.equal(k,k1) and torch.equal(v,v1)
