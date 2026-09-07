"""Scoring invariants.

The distractor-aware `clean_hit` is the headline metric for every stage, so its failure
modes are worth pinning down explicitly.
"""

from __future__ import annotations

import pytest

from ccl.corpus import Fact
from ccl.metrics import aggregate, norm_exact, norm_semantic, score


def _fact(cls="identifier", answer="QZ-8471-BX", distractor="QZ-8472-BX"):
    return Fact(
        fact_id="t-0",
        fact_class=cls,
        question="q?",
        answer=answer,
        answer_hint="h",
        distractor_answer=distractor,
        support_chunks=[0],
        distractor_chunks=[1],
    )


# -- normalizers -----------------------------------------------------------------------

def test_semantic_normalizer_is_case_and_punctuation_insensitive():
    assert norm_semantic("Blue.") == norm_semantic("  blue ")


def test_exact_normalizer_preserves_case_and_structure():
    assert norm_exact("QZ-8471-BX") == "QZ-8471-BX"
    assert norm_exact("qz-8471-bx") != norm_exact("QZ-8471-BX")


def test_exact_normalizer_drops_thousands_separators_only():
    assert norm_exact("48,317") == "48317"
    assert norm_exact("a,bcd") == "a,bcd"


# -- exact classes ---------------------------------------------------------------------

def test_exact_hit_requires_exact_characters():
    f = _fact()
    assert score(f, "QZ-8471-BX")["clean_hit"] is True
    assert score(f, "QZ-8471-BY")["clean_hit"] is False
    assert score(f, "qz-8471-bx")["clean_hit"] is False


def test_answer_embedded_in_a_sentence_still_counts():
    assert score(_fact(), "The assembly is QZ-8471-BX.")["clean_hit"] is True


def test_emitting_the_distractor_is_never_clean():
    f = _fact()
    assert score(f, "QZ-8472-BX")["clean_hit"] is False


def test_emitting_both_gold_and_distractor_is_not_clean():
    """Hedging across both values is not knowledge, and a bare containment score would
    reward it."""
    f = _fact()
    row = score(f, "Either QZ-8471-BX or QZ-8472-BX.")
    assert row["exact_hit"] is True
    assert row["distractor_hit"] is True
    assert row["clean_hit"] is False


def test_number_formatting_difference_still_matches():
    f = _fact(cls="number", answer="48,317", distractor="21,971")
    assert score(f, "48317")["clean_hit"] is True


# -- semantic class --------------------------------------------------------------------

def test_semantic_matching_is_word_bounded():
    """Substring matching would score 'blue' as a hit inside 'bluebell'."""
    f = _fact(cls="semantic", answer="blue", distractor="amber")
    assert score(f, "blue")["clean_hit"] is True
    assert score(f, "The bluebell was in bloom.")["clean_hit"] is False


def test_semantic_class_is_not_scored_by_exact_match():
    f = _fact(cls="semantic", answer="blue", distractor="amber")
    assert score(f, "Blue.")["clean_hit"] is True


def test_unknown_answer_is_a_miss():
    for cls, ans, dis in [
        ("identifier", "QZ-8471-BX", "QZ-8472-BX"),
        ("semantic", "blue", "amber"),
    ]:
        assert score(_fact(cls, ans, dis), "UNKNOWN")["clean_hit"] is False


# -- aggregation -----------------------------------------------------------------------

def test_aggregate_reports_overall_and_per_class():
    rows = [
        score(_fact(), "QZ-8471-BX"),
        score(_fact(cls="semantic", answer="blue", distractor="amber"), "amber"),
    ]
    for r in rows:
        r["rank_margin"] = 1.0
        r["answer_logprob"] = -0.5
    agg = aggregate(rows)
    assert agg["overall"]["n"] == 2
    assert agg["overall"]["clean_hit"] == pytest.approx(0.5)
    assert set(agg["by_class"]) == {"identifier", "semantic"}
    assert agg["by_class"]["identifier"]["clean_hit"] == 1.0
    assert agg["by_class"]["semantic"]["clean_hit"] == 0.0
    assert agg["overall"]["rank_margin"] == pytest.approx(1.0)


def test_aggregate_handles_empty_input():
    assert aggregate([])["overall"] == {"n": 0}
