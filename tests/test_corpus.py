"""Corpus invariants.

These guard the things that, if quietly broken, would make every downstream number
meaningless: draw disjointness, determinism, distractor placement, and the promise that
compilation never sees a question.
"""

from __future__ import annotations

import pytest

from ccl.corpus import (
    ANSWER_HINTS,
    EXACT_CLASSES,
    FACT_CLASSES,
    POOLS,
    all_values,
    approx_token_count,
    content_id,
    generate_corpus,
    generate_document,
    normalize_chunk_bytes,
)


def _doc(draw="A", seed=1, idx=0, tokens=1024, **kw):
    return generate_document(
        draw=draw,
        seed=seed,
        doc_index=idx,
        target_tokens=tokens,
        count_tokens=approx_token_count,
        **kw,
    )


# -- determinism -----------------------------------------------------------------------

def test_generation_is_deterministic():
    a, b = _doc(), _doc()
    assert a.text == b.text
    assert [f.answer for f in a.facts] == [f.answer for f in b.facts]
    assert [c.content_id for c in a.chunks] == [c.content_id for c in b.chunks]


def test_different_seed_gives_different_document():
    assert _doc(seed=1).text != _doc(seed=2).text


def test_different_doc_index_gives_different_document():
    assert _doc(idx=0).text != _doc(idx=1).text


# -- draw disjointness (anti-leakage, docs/EXPERIMENT.md section 8) ---------------------

def test_draw_pools_are_disjoint():
    a, b = POOLS["A"], POOLS["B"]
    for field in (
        "given", "surname", "color", "sysroot", "facility", "role", "place",
        "idprefix", "pathroot", "host", "audit", "quarter", "unit",
    ):
        overlap = set(getattr(a, field)) & set(getattr(b, field))
        assert not overlap, f"pool {field!r} overlaps between draws: {overlap}"
    assert a.year_hi < b.year_lo
    assert a.num_hi < b.num_lo


def test_generated_values_are_disjoint_across_draws():
    lengths = [1024, 2048]
    va = all_values(
        generate_corpus(
            draw="A", seed=11, lengths=lengths, count_tokens=approx_token_count, n_docs=4
        )
    )
    vb = all_values(
        generate_corpus(
            draw="B", seed=977, lengths=lengths, count_tokens=approx_token_count, n_docs=4
        )
    )
    assert not (va & vb), f"held-out draw shares values with draw A: {sorted(va & vb)[:5]}"


# -- structural invariants -------------------------------------------------------------

def test_all_fact_classes_present_and_hinted():
    doc = _doc()
    assert {f.fact_class for f in doc.facts} == set(FACT_CLASSES)
    for f in doc.facts:
        assert f.answer_hint == ANSWER_HINTS[f.fact_class]


def test_every_fact_has_a_distinct_distractor():
    doc = _doc(tokens=4096)
    for f in doc.facts:
        assert f.answer != f.distractor_answer, f.fact_id
        assert f.distractor_answer, f.fact_id


def test_gold_value_appears_in_its_support_chunks():
    doc = _doc(tokens=4096)
    for f in doc.facts:
        joined = " ".join(doc.chunks[i].text for i in f.support_chunks)
        assert f.answer in joined, f"{f.fact_id}: gold missing from support chunks"


def test_distractor_lives_outside_support_for_single_chunk_facts():
    doc = _doc(tokens=4096)
    for f in doc.facts:
        if f.fact_class == "composition":
            continue
        assert f.support_chunks != f.distractor_chunks, (
            f"{f.fact_id}: distractor shares the support chunk, so a retrieval-based "
            "condition could never be fooled by it"
        )


def test_composition_facts_span_two_separated_chunks():
    doc = _doc(tokens=8192)
    comp = [f for f in doc.facts if f.fact_class == "composition"]
    assert comp
    for f in comp:
        assert len(f.support_chunks) == 2, f.fact_id
        assert f.support_chunks[1] - f.support_chunks[0] >= 2, (
            f"{f.fact_id}: composition halves are adjacent, so the join is local"
        )


def test_chunks_are_nonempty_and_roughly_budgeted():
    doc = _doc(tokens=8192, chunk_tokens=256)
    assert len(doc.chunks) == 32
    for c in doc.chunks:
        assert c.text.strip()
        assert c.n_tokens > 0


def test_questions_never_contain_the_answer():
    """Compilation is a pure function of the chunk; a leaky question would let a
    downstream condition score well with no context at all."""
    for tokens in (1024, 4096):
        for f in _doc(tokens=tokens).facts:
            assert f.answer.lower() not in f.question.lower(), f.fact_id


def test_exact_classes_cover_everything_but_semantic():
    assert set(FACT_CLASSES) - EXACT_CLASSES == {"semantic"}


# -- content addressing ----------------------------------------------------------------

def test_content_id_is_stable_and_normalizing():
    assert content_id("hello world") == content_id("hello world  \n")
    assert content_id("hello world") != content_id("hello  world")
    assert len(content_id("x")) == 64


def test_identical_chunks_deduplicate():
    """Stage 5 depends on this: the same bytes must produce the same page ID."""
    a, b = _doc(idx=0), _doc(idx=0)
    assert {c.content_id for c in a.chunks} == {c.content_id for c in b.chunks}


def test_normalize_chunk_bytes_is_idempotent():
    raw = "  a \r\n b  \n\n"
    once = normalize_chunk_bytes(raw)
    assert normalize_chunk_bytes(once.decode()) == once


@pytest.mark.parametrize("tokens", [1024, 2048, 4096])
def test_chunk_count_tracks_target_length(tokens):
    assert len(_doc(tokens=tokens, chunk_tokens=256).chunks) == tokens // 256
