"""Positional guards for pre-RoPE storage and composition.

These are the tests that decide whether anything in Stage 2b/3 means what it claims.
Four failure modes are guarded explicitly, because each of them produces a *plausible
looking* system that is silently wrong:

* keys stored post-RoPE — the page is permanently dated to its extraction position;
* RoPE never re-applied — every page believes it sits at position 0;
* RoPE applied twice — positions doubled, and no error is raised anywhere;
* pages not actually re-positioned per offset — composition is a lie.

The shape-level tests run without weights. The end-to-end identity test is `slow`.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from ccl.rope import apply_rope_to_keys  # noqa: E402

HEAD_DIM = 8


def _rope_tables(positions, theta: float = 1_000_000.0, dim: int = HEAD_DIM):
    """cos/sin tables matching transformers' layout: [1, S, dim] with freqs duplicated."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = torch.outer(torch.as_tensor(positions, dtype=torch.float), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def _keys(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 2, n, HEAD_DIM, generator=g)


# -- the rotation itself ----------------------------------------------------------------

def test_rope_at_position_zero_is_identity():
    """Position 0 has zero angle, so rotation must be a no-op. If this fails, the cos/sin
    layout is wrong and every other positional result is meaningless."""
    k = _keys(4)
    cos, sin = _rope_tables([0, 0, 0, 0])
    assert torch.allclose(apply_rope_to_keys(k, cos, sin), k, atol=1e-6)


def test_rope_preserves_norm():
    """Rotation is orthogonal: it moves phase, never magnitude."""
    k = _keys(6)
    cos, sin = _rope_tables(range(6))
    out = apply_rope_to_keys(k, cos, sin)
    assert torch.allclose(out.norm(dim=-1), k.norm(dim=-1), atol=1e-5)


def test_rope_at_different_positions_differs():
    """Guards 'positions were never actually applied'."""
    k = _keys(1)
    a = apply_rope_to_keys(k, *_rope_tables([0]))
    b = apply_rope_to_keys(k, *_rope_tables([17]))
    assert not torch.allclose(a, b, atol=1e-4)


def test_double_rope_is_detectably_different():
    """Applying RoPE twice at p equals applying it once at 2p, and must not be mistaken
    for the correct result. Nothing in the stack raises on a double rotation, so this is
    the only thing standing between us and a silent positional bug."""
    k = _keys(3)
    once = apply_rope_to_keys(k, *_rope_tables([5, 5, 5]))
    twice = apply_rope_to_keys(once, *_rope_tables([5, 5, 5]))
    direct = apply_rope_to_keys(k, *_rope_tables([10, 10, 10]))
    assert torch.allclose(twice, direct, atol=1e-5), "double rotation should compose"
    assert not torch.allclose(twice, once, atol=1e-3), "double rotation must be detectable"


def test_composition_of_rotations_is_additive_in_position():
    """Rotating by a then b equals rotating by a+b — the property that lets a stored
    pre-RoPE page be placed at an arbitrary offset."""
    k = _keys(2)
    a, b = 7, 11
    step = apply_rope_to_keys(apply_rope_to_keys(k, *_rope_tables([a, a])), *_rope_tables([b, b]))
    direct = apply_rope_to_keys(k, *_rope_tables([a + b, a + b]))
    assert torch.allclose(step, direct, atol=1e-5)


def test_relative_angle_between_two_placements_matches_offset():
    """The same page placed at two offsets differs by exactly the rotation between them.
    This is what 'the page is not baked to its extraction position' means concretely."""
    k = _keys(1, seed=3)
    at10 = apply_rope_to_keys(k, *_rope_tables([10]))
    at25 = apply_rope_to_keys(k, *_rope_tables([25]))
    rotated = apply_rope_to_keys(at10, *_rope_tables([15]))
    assert torch.allclose(rotated, at25, atol=1e-5)


def test_first_frequency_pair_rotates_by_the_expected_angle():
    """Pin the actual angle, not just self-consistency: a self-consistent but wrongly
    scaled RoPE would pass every test above."""
    k = torch.zeros(1, 1, 1, HEAD_DIM)
    k[0, 0, 0, 0] = 1.0  # unit vector in the first coordinate of pair 0
    p = 3
    out = apply_rope_to_keys(k, *_rope_tables([p]))
    half = HEAD_DIM // 2
    assert out[0, 0, 0, 0].item() == pytest.approx(math.cos(p), abs=1e-5)
    assert out[0, 0, 0, half].item() == pytest.approx(math.sin(p), abs=1e-5)


# -- block layout -----------------------------------------------------------------------

def test_block_spans_partition_the_chunk_exactly():
    from ccl.compressor import block_spans

    for n in (16, 100, 259):
        for r in (1, 2, 4, 8, 16):
            spans = block_spans(n, r)
            assert spans[0][0] == 0
            assert spans[-1][1] == n
            for a, b in zip(spans, spans[1:]):
                assert a[1] == b[0], "blocks must tile without gap or overlap"


@pytest.mark.parametrize("ratio,expected", [(1, 259), (2, 130), (4, 65), (8, 33), (16, 17)])
def test_memory_token_count_tracks_ratio(ratio, expected):
    from ccl.compressor import block_spans

    assert len(block_spans(259, ratio)) == expected


def test_extraction_mask_enforces_c2kv_constraints():
    from ccl.compressor import block_spans, extraction_mask

    n, n_sink = 32, 4
    spans = block_spans(n, 4)
    mask = extraction_mask(spans, n, n_sink, device="cpu")
    m = len(spans)
    assert mask.shape == (m, n + m)
    for j, (lo, hi) in enumerate(spans):
        # block-local: own block visible
        assert mask[j, lo:hi].all()
        # sink block always visible
        assert mask[j, :n_sink].all()
        # nothing else from the chunk
        outside = [
            i for i in range(n) if not (lo <= i < hi) and i >= n_sink
        ]
        assert not mask[j, outside].any()
        # causal accumulation over memory tokens
        assert mask[j, n : n + j + 1].all()
        assert not mask[j, n + j + 1 :].any()


def test_extraction_mask_never_leaves_a_row_empty():
    """An all-masked row would produce NaN through softmax."""
    from ccl.compressor import block_spans, extraction_mask

    for n in (8, 64, 259):
        for r in (1, 4, 16):
            spans = block_spans(n, r)
            mask = extraction_mask(spans, n, 0, device="cpu")
            assert mask.any(dim=1).all()


# -- end-to-end identity ----------------------------------------------------------------

@pytest.mark.slow
def test_prerope_identity_reproduces_native_prefill():
    """Capture the target's own pre-RoPE KV, recompose at the same offset, and check the
    result matches an ordinary prefill within bf16 accumulation noise.

    Measured reference: arbitrary split points of the *identical* computation differ by
    up to ~1.0 on logits of scale ~55, so the tolerance is set there. A double-RoPE bug
    produces ~35, i.e. this test discriminates by a wide margin.
    """
    from ccl.corpus import load_corpus
    from ccl.rope import PreRopeCapture, cache_from_layers, compose_pages
    from ccl.target import TargetModel

    docs = load_corpus("results/raw/corpus/draw_A.json")
    doc = next(d for d in docs if d.target_tokens == 1024)
    fact = doc.facts[1]
    tm = TargetModel("Qwen/Qwen3-4B")
    parts = tm.split_prompt4(fact.question, doc.text, fact.answer_hint)
    assert parts is not None
    head, body, tail, suffix = parts
    prefix = torch.cat([tm.encode(head), tm.encode(body), tm.encode(tail)], dim=1)
    s = tm.encode(suffix)

    ref, _ = tm.prefill(tm.encode(tm.build_prompt(fact.question, doc.text, fact.answer_hint)))

    with PreRopeCapture(tm.model) as cap, torch.no_grad():
        tm.prefill(prefix)
        page = cap.stacked()
    layers, nxt = compose_pages(tm.model, [page], start=0)
    cache = cache_from_layers(tm.model, layers)
    pos = torch.arange(nxt, nxt + s.shape[1], device=tm.device)
    got, _ = tm.prefill(s, cache, position_ids=pos, cache_position=pos.clone())

    assert int(ref.argmax()) == int(got.argmax())
    assert float((ref.float() - got.float()).abs().max()) < 2.0
