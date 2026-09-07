"""Training-example construction invariants.

The two that matter scientifically: an example must always contain the evidence for its
own answer (otherwise we teach the compressor to hallucinate), and the compressor must
never be handed a question.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch")

from ccl.corpus import approx_token_count, generate_corpus  # noqa: E402
from ccl.train import build_examples  # noqa: E402


@pytest.fixture(scope="module")
def docs():
    return generate_corpus(
        draw="A", seed=11, lengths=[2048], count_tokens=approx_token_count, n_docs=3
    )


def test_every_example_contains_its_own_support_chunks(docs):
    """Training on a context that lacks the answer would teach the model to invent one."""
    ex = build_examples(docs, random.Random(0))
    assert ex
    for e in ex:
        assert set(e.fact.support_chunks).issubset(set(e.chunk_ids)), e.fact.fact_id


def test_examples_cover_every_fact(docs):
    ex = build_examples(docs, random.Random(0))
    assert len(ex) == sum(len(d.facts) for d in docs)
    assert {e.fact.fact_id for e in ex} == {f.fact_id for d in docs for f in d.facts}


def test_chunk_ids_are_sorted_unique_and_in_range(docs):
    for e in build_examples(docs, random.Random(0)):
        assert e.chunk_ids == sorted(set(e.chunk_ids))
        assert all(0 <= i < len(e.doc.chunks) for i in e.chunk_ids)


def test_chunk_count_respects_bounds(docs):
    for e in build_examples(docs, random.Random(0), min_chunks=2, max_chunks=4):
        # A composition fact needs 2 support chunks, so the floor can exceed min_chunks;
        # the ceiling is what must hold.
        assert len(e.chunk_ids) >= 2
        assert len(e.chunk_ids) <= max(4, len(e.fact.support_chunks))


def test_examples_include_filler_beyond_the_supporting_chunks(docs):
    """The compressor must encode content it will not be asked about — otherwise the
    task degenerates into 'compress exactly the answer'."""
    ex = build_examples(docs, random.Random(0), min_chunks=3, max_chunks=5)
    assert any(len(e.chunk_ids) > len(e.fact.support_chunks) for e in ex)


def test_build_examples_is_deterministic_given_a_seed(docs):
    a = build_examples(docs, random.Random(7))
    b = build_examples(docs, random.Random(7))
    assert [e.chunk_ids for e in a] == [e.chunk_ids for e in b]


def test_extractor_signature_has_no_question_parameter():
    """Structural guarantee, not a convention: `extract` takes a chunk and a ratio, so a
    question-conditioned shortcut is unrepresentable rather than merely discouraged."""
    import inspect

    from ccl.compressor import MemoryExtractor

    params = set(inspect.signature(MemoryExtractor.extract).parameters)
    assert params == {"self", "chunk_ids", "ratio"}
    compile_params = set(inspect.signature(MemoryExtractor.compile_chunks).parameters)
    assert compile_params == {"self", "chunk_id_list", "ratio"}
