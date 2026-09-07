"""Target-model plumbing that must be right before any compressed state is spliced in.

Nothing here loads weights. The geometry and prompt-splitting checks use the cached
config/tokenizer only; anything needing a forward pass is marked `slow`.
"""

from __future__ import annotations

import pytest

MODEL = "Qwen/Qwen3-4B"


@pytest.fixture(scope="module")
def geometry():
    pytest.importorskip("torch")
    from ccl.target import read_geometry

    try:
        return read_geometry(MODEL)
    except Exception as exc:  # pragma: no cover - no cached model on this machine
        pytest.skip(f"{MODEL} config unavailable: {exc}")


def test_geometry_matches_documented_qwen3_4b(geometry):
    """AGENTS.md quotes these; if the checkpoint ever changes, fail loudly here."""
    assert geometry.hidden_size == 2560
    assert geometry.num_hidden_layers == 36
    assert geometry.num_attention_heads == 32
    assert geometry.num_key_value_heads == 8
    assert geometry.head_dim == 128
    assert geometry.gqa_ratio == 4.0


def test_rope_theta_is_resolved_not_defaulted(geometry):
    """transformers 5.x moved rope_theta into rope_scaling. Reading 0.0 (or a 4.x-era
    default of 10_000) would corrupt every re-applied position in Stage 3."""
    assert geometry.rope_theta == 1_000_000.0


def test_kv_bytes_per_token_matches_geometry(geometry):
    expected = 2 * 36 * 8 * 128 * 2  # K and V, all layers, bf16
    assert geometry.kv_bytes_per_token == expected == 147_456


def test_geometry_serializes_derived_fields(geometry):
    d = geometry.to_json()
    assert d["gqa_ratio"] == 4.0
    assert d["kv_bytes_per_token"] == 147_456


# -- prompt construction ----------------------------------------------------------------

@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{MODEL} tokenizer unavailable: {exc}")


class _StubTarget:
    """`TargetModel`'s prompt logic without loading 8 GB of weights."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    _chat = property(lambda self: None)


def _make_stub(tokenizer):
    from ccl.target import TargetModel

    stub = _StubTarget.__new__(TargetModel)
    stub.tokenizer = tokenizer
    return stub


def test_noctx_prompt_contains_no_document(tok):
    p = _make_stub(tok).build_prompt("What color?", None, "a color")
    assert "<document>" not in p
    assert "What color?" in p


def test_native_prompt_embeds_the_document_and_hint(tok):
    p = _make_stub(tok).build_prompt("What color?", "DOCBODY", "a color")
    assert "<document>" in p and "DOCBODY" in p
    assert "(a color)" in p


def test_thinking_is_disabled_so_answers_stay_terse(tok):
    p = _make_stub(tok).build_prompt("q?", None, None)
    assert "<think>\n\n</think>" in p


def test_split_prompt_is_an_exact_token_partition(tok):
    """Prefix reuse is only valid if prefix+suffix tokenizes as the concatenation of the
    two. If BPE merges across the seam, `split_prompt` must return None rather than
    silently reusing a misaligned cache."""
    stub = _make_stub(tok)
    doc = "### Section 000\nSome document body with facts in it. " * 20
    split = stub.split_prompt("What color?", doc, "a color")
    assert split is not None
    prefix, suffix = split
    full = stub.build_prompt("What color?", doc, "a color")
    assert prefix + suffix == full
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    assert enc(prefix) + enc(suffix) == enc(full)


def test_split_prompt_prefix_is_shared_across_questions(tok):
    """The whole point: one document prefill serves every question about that document."""
    stub = _make_stub(tok)
    doc = "### Section 000\nBody. " * 30
    a = stub.split_prompt("First question?", doc, "a color")
    b = stub.split_prompt("Second, different question?", doc, "a date")
    assert a is not None and b is not None
    assert a[0] == b[0]
    assert a[1] != b[1]


# -- cache helper -----------------------------------------------------------------------

@pytest.mark.slow
def test_crop_to_restores_cache_length():
    torch = pytest.importorskip("torch")
    from transformers import AutoConfig, DynamicCache

    from ccl.target import crop_to

    cfg = AutoConfig.from_pretrained(MODEL)
    cache = DynamicCache(config=cfg)
    for layer in range(cfg.num_hidden_layers):
        k = torch.zeros(1, 8, 10, 128)
        cache.update(k, k.clone(), layer)
    assert cache.get_seq_length() == 10
    crop_to(cache, 4)
    assert cache.get_seq_length() == 4
    crop_to(cache, 99)  # never grows
    assert cache.get_seq_length() == 4
