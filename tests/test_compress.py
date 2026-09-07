"""Training-free compression and control invariants.

The controls are only worth running if they do what they claim. `RANDOM` must destroy
information without going off-distribution, `compress_cache` must touch only the
document span, and a ratio of 1.0 must be an exact no-op — otherwise a measured
"compression cost" is partly just harness damage.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ccl.compress import (  # noqa: E402
    budget_text,
    compress_cache,
    keep_indices,
    randomize_cache,
    shuffle_cache_blocks,
    shuffle_document_text,
)


class _FakeLayer:
    def __init__(self, n: int):
        # Distinct per position so permutations and selections are detectable.
        base = torch.arange(n, dtype=torch.float32).reshape(1, 1, n, 1)
        self.keys = base.repeat(1, 2, 1, 4).clone()
        self.values = (base + 1000).repeat(1, 2, 1, 4).clone()


class _FakeCache:
    def __init__(self, n: int, n_layers: int = 3):
        self.layers = [_FakeLayer(n) for _ in range(n_layers)]

    def get_seq_length(self) -> int:
        return self.layers[0].keys.shape[2]


def _positions(cache: _FakeCache) -> list[int]:
    return cache.layers[0].keys[0, 0, :, 0].int().tolist()


# -- keep_indices ------------------------------------------------------------------

@pytest.mark.parametrize("ratio", [2.0, 4.0, 8.0, 16.0])
def test_stride_keeps_about_one_in_ratio(ratio):
    kept = keep_indices(1000, ratio, "stride")
    assert len(kept) == pytest.approx(1000 / ratio, rel=0.05)
    assert kept == sorted(set(kept))


def test_ratio_one_keeps_everything():
    """A ratio-1 run must be an exact no-op, so it can serve as the identity check."""
    assert keep_indices(500, 1.0, "stride") == list(range(500))


def test_stride_is_uniformly_spread_not_clustered():
    kept = keep_indices(1000, 10.0, "stride")
    assert min(kept) < 50 and max(kept) > 950


def test_sink_always_retains_the_earliest_positions():
    """Attention-sink positions absorb disproportionate attention mass regardless of
    content, so dropping them is unusually damaging."""
    kept = keep_indices(1000, 8.0, "sink", n_sink=4)
    assert set(range(4)).issubset(set(kept))


def test_sink_and_stride_have_comparable_budgets():
    a = keep_indices(1000, 8.0, "stride")
    b = keep_indices(1000, 8.0, "sink", n_sink=4)
    assert abs(len(a) - len(b)) <= 5


def test_head_and_tail_are_contiguous_and_disjoint_in_placement():
    h = keep_indices(100, 4.0, "head")
    t = keep_indices(100, 4.0, "tail")
    assert h == list(range(len(h)))
    assert t == list(range(100 - len(t), 100))


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        keep_indices(10, 2.0, "telepathy")


def test_empty_input_is_handled():
    assert keep_indices(0, 4.0, "stride") == []


# -- compress_cache ----------------------------------------------------------------

def test_compress_touches_only_the_document_span():
    cache = _FakeCache(20)
    survivors = compress_cache(cache, doc_start=5, doc_end=15, kept=[0, 2, 4])
    assert survivors == [0, 1, 2, 3, 4, 5, 7, 9, 15, 16, 17, 18, 19]
    assert _positions(cache) == survivors
    assert cache.get_seq_length() == 13


def test_compress_with_full_keep_is_identity():
    cache = _FakeCache(20)
    before = _positions(cache)
    compress_cache(cache, 5, 15, list(range(10)))
    assert _positions(cache) == before


def test_compress_applies_to_every_layer():
    cache = _FakeCache(20, n_layers=3)
    compress_cache(cache, 5, 15, [0, 1])
    assert {lay.keys.shape[2] for lay in cache.layers} == {12}
    assert {lay.values.shape[2] for lay in cache.layers} == {12}


def test_compress_keeps_keys_and_values_aligned():
    cache = _FakeCache(20)
    compress_cache(cache, 5, 15, [0, 3])
    k = cache.layers[0].keys[0, 0, :, 0]
    v = cache.layers[0].values[0, 0, :, 0]
    assert torch.equal(v - k, torch.full_like(k, 1000.0))


# -- RANDOM control ----------------------------------------------------------------

def test_random_replaces_document_span_only():
    cache = _FakeCache(20)
    randomize_cache(cache, 5, 15, seed=0)
    pos = _positions(cache)
    assert pos[:5] == [0, 1, 2, 3, 4]
    assert pos[15:] == [15, 16, 17, 18, 19]
    assert pos[5:15] != [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_random_is_moment_matched_not_merely_zeroed():
    """Off-distribution noise would be easy for the model to ignore, making the control
    weaker than it looks. Matched moments carry no information but look like cache."""
    cache = _FakeCache(200)
    span_before = cache.layers[0].keys[:, :, 20:180, :].clone()
    randomize_cache(cache, 20, 180, seed=0)
    span_after = cache.layers[0].keys[:, :, 20:180, :]
    assert span_after.mean().item() == pytest.approx(span_before.mean().item(), abs=2.0)
    assert span_after.std().item() == pytest.approx(span_before.std().item(), rel=0.25)


def test_random_is_deterministic_for_a_given_seed():
    a, b = _FakeCache(20), _FakeCache(20)
    randomize_cache(a, 5, 15, seed=7)
    randomize_cache(b, 5, 15, seed=7)
    assert torch.equal(a.layers[0].keys, b.layers[0].keys)


# -- shuffle controls ---------------------------------------------------------------

def test_kv_order_shuffle_is_a_permutation_of_the_document_span():
    cache = _FakeCache(24)
    shuffle_cache_blocks(cache, 4, 20, block=4, seed=1)
    pos = _positions(cache)
    assert pos[:4] == [0, 1, 2, 3]
    assert pos[20:] == [20, 21, 22, 23]
    assert sorted(pos[4:20]) == list(range(4, 20))


def test_kv_order_shuffle_preserves_blocks_intact():
    cache = _FakeCache(24)
    shuffle_cache_blocks(cache, 4, 20, block=4, seed=1)
    mid = _positions(cache)[4:20]
    for i in range(0, 16, 4):
        block = mid[i : i + 4]
        assert block == list(range(block[0], block[0] + 4))


def test_text_shuffle_preserves_every_chunk():
    chunks = [f"chunk {i}" for i in range(10)]
    out = shuffle_document_text(chunks, seed=0)
    assert sorted(out.split("\n\n")) == sorted(chunks)


# -- BUDGET control -----------------------------------------------------------------

@pytest.mark.parametrize("ratio", [2.0, 4.0, 8.0])
def test_budget_keeps_about_one_in_ratio_chunks(ratio):
    chunks = [f"c{i}" for i in range(64)]
    kept = budget_text(chunks, ratio).split("\n\n")
    assert len(kept) == pytest.approx(64 / ratio, rel=0.1)


def test_budget_spreads_across_the_document_rather_than_truncating():
    """A prefix-truncation BUDGET would be a strawman: it loses every late fact by
    construction."""
    chunks = [f"c{i}" for i in range(64)]
    kept = budget_text(chunks, 8.0).split("\n\n")
    assert kept[0] == "c0"
    assert int(kept[-1][1:]) > 48


def test_budget_preserves_document_order():
    chunks = [f"c{i}" for i in range(64)]
    kept = [int(c[1:]) for c in budget_text(chunks, 4.0).split("\n\n")]
    assert kept == sorted(kept)
