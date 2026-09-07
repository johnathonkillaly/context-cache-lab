"""Scoring.

Two normalizers, deliberately different:

* `norm_semantic` is permissive (case-folded, punctuation-stripped) because a semantic
  answer like "blue" can legitimately arrive inside a sentence.
* `norm_exact` preserves case and internal punctuation because an identifier, path, URL
  or digest that differs by one character is *wrong*, and the whole point of the exact
  fact classes is to detect exactly that.

Every exact score is also reported in a `clean` variant that requires the model to have
avoided the matched distractor. A bare containment score is inflated by format-guessing;
the clean variant is the one to quote.
"""

from __future__ import annotations

import re
import unicodedata

from .corpus import EXACT_CLASSES, Fact

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def norm_semantic(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip()


def norm_exact(s: str) -> str:
    """Whitespace-normalized, case- and punctuation-preserving.

    Thousands separators are dropped so that `48,317` and `48317` compare equal; that is
    a rendering difference, not a factual one.
    """
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return _WS.sub(" ", s).strip()


def _contains_semantic(pred: str, gold: str) -> bool:
    p, g = norm_semantic(pred), norm_semantic(gold)
    if not g:
        return False
    return re.search(rf"(?<!\w){re.escape(g)}(?!\w)", p) is not None


def _contains_exact(pred: str, gold: str) -> bool:
    p, g = norm_exact(pred), norm_exact(gold)
    return bool(g) and g in p


def score(fact: Fact, prediction: str) -> dict:
    """Score one prediction against one fact.

    Returns both the permissive and strict views so that later analysis can choose,
    and so that `semantic_hit` on an exact class stays visible (a model that gets the
    right *kind* of answer but the wrong characters is a different failure from a model
    that has lost the fact entirely - that distinction is the heart of the
    "what breaks first" question in docs/EXPERIMENT.md).
    """
    is_exact_class = fact.fact_class in EXACT_CLASSES

    semantic_hit = _contains_semantic(prediction, fact.answer)
    exact_hit = _contains_exact(prediction, fact.answer)
    distractor_hit = (
        _contains_exact(prediction, fact.distractor_answer)
        if is_exact_class
        else _contains_semantic(prediction, fact.distractor_answer)
    )

    primary = exact_hit if is_exact_class else semantic_hit
    return {
        "fact_id": fact.fact_id,
        "fact_class": fact.fact_class,
        "is_exact_class": is_exact_class,
        "semantic_hit": semantic_hit,
        "exact_hit": exact_hit,
        "distractor_hit": distractor_hit,
        # The headline number: correct, and not fooled by the matched distractor.
        "clean_hit": bool(primary and not distractor_hit),
        "prediction": prediction,
        "gold": fact.answer,
        "distractor": fact.distractor_answer,
    }


def aggregate(rows: list[dict]) -> dict:
    """Aggregate scored rows overall and per fact class."""
    def _agg(subset: list[dict]) -> dict:
        n = len(subset)
        if n == 0:
            return {"n": 0}
        out = {"n": n}
        for k in ("semantic_hit", "exact_hit", "distractor_hit", "clean_hit"):
            out[k] = sum(bool(r[k]) for r in subset) / n
        margins = [r["rank_margin"] for r in subset if r.get("rank_margin") is not None]
        if margins:
            out["rank_margin"] = sum(margins) / len(margins)
        lps = [r["answer_logprob"] for r in subset if r.get("answer_logprob") is not None]
        if lps:
            out["answer_logprob"] = sum(lps) / len(lps)
        return out

    classes = sorted({r["fact_class"] for r in rows})
    return {
        "overall": _agg(rows),
        "by_class": {c: _agg([r for r in rows if r["fact_class"] == c]) for c in classes},
    }
