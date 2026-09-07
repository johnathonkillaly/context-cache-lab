"""Deterministic synthetic long-context corpus.

Why synthetic: we need exact ground truth, controllable fact placement, matched
distractors, and a guarantee that the facts are absent from the target model's
pretraining data. A model answering from parametric memory would silently invalidate
every downstream comparison, so the NOCTX control has to be able to reach the floor.

Two independent draws, A and B. Draw A is used for development and any training. Draw B
is held out and evaluated once against the frozen criteria in docs/EXPERIMENT.md. The
value pools of A and B are disjoint **by construction**, and tests assert it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

FACT_CLASSES = (
    "semantic",
    "identifier",
    "number",
    "date",
    "hash",
    "path",
    "url",
    "proper_noun",
    "composition",
)

#: Classes scored by exact string match rather than loose containment.
EXACT_CLASSES = frozenset(
    {"identifier", "number", "date", "hash", "path", "url", "proper_noun", "composition"}
)


# --------------------------------------------------------------------------------------
# Content identity (Stage 5 uses this; defined here because it is a property of a chunk)
# --------------------------------------------------------------------------------------

def normalize_chunk_bytes(text: str) -> bytes:
    """Normalized bytes used for content addressing.

    NFC + trailing-whitespace strip + LF line endings, so that cosmetically different
    but semantically identical chunks deduplicate. Deliberately conservative: it must
    never merge two chunks whose *tokens* would differ.
    """
    norm = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in norm.split("\n")]
    return "\n".join(lines).strip().encode("utf-8")


def content_id(text: str) -> str:
    """Stable content identifier for a chunk.

    NOTE: this is an **ID only**. It is not a semantic embedding and carries no
    similarity structure. Retrieval uses `r_i` (Stage 6), never this.
    """
    return hashlib.sha256(normalize_chunk_bytes(text)).hexdigest()


# --------------------------------------------------------------------------------------
# Disjoint value pools
# --------------------------------------------------------------------------------------

_GIVEN_A = ["Halvard", "Miriam", "Tobias", "Ingrid", "Casimir", "Rowena", "Anselm", "Dagny"]
_GIVEN_B = ["Sabine", "Lorcan", "Petra", "Emeric", "Yvonne", "Bastian", "Ottilie", "Renzo"]
_SUR_A = ["Nieminen", "Ashgrove", "Vantol", "Okonkwo", "Brannigan", "Delacroix", "Sorley"]
_SUR_B = ["Kastellan", "Merriwether", "Vasquez", "Thornbury", "Aldabra", "Ferreiro", "Quint"]

_COLOR_A = ["blue", "amber", "olive", "crimson", "slate", "teal"]
_COLOR_B = ["magenta", "ochre", "indigo", "russet", "chartreuse", "cobalt"]

_SYSROOT_A = ["Mirren", "Calder", "Vestry", "Harlow", "Ordell"]
_SYSROOT_B = ["Pelagic", "Sundry", "Kirtland", "Brackwell", "Ninefold"]

_FACILITY_A = ["Longmere", "Ashcombe", "Draywell", "Fennhollow", "Ravensgate"]
_FACILITY_B = ["Stonebarrow", "Halloway", "Grimsdale", "Wetherby", "Corvane"]

_ROLE_A = ["site reliability lead", "compliance registrar", "logistics coordinator"]
_ROLE_B = ["procurement auditor", "systems archivist", "field operations warden"]

_PLACE_A = ["Ashcombe Junction", "Longmere Bridge", "Draywell Crossing"]
_PLACE_B = ["Stonebarrow Cut", "Halloway Reach", "Grimsdale Ford"]

_IDPREFIX_A = ["QZ", "RT", "KP", "MB", "XD"]
_IDPREFIX_B = ["VL", "HN", "GS", "WF", "CJ"]

_PATHROOT_A = ["/var/opt/mirren", "/srv/calder", "/opt/vestry"]
_PATHROOT_B = ["/var/lib/pelagic", "/srv/kirtland", "/opt/brackwell"]

_HOST_A = ["depot.example.net", "relay.example.org"]
_HOST_B = ["vault.example.com", "gateway.example.io"]

_AUDIT_A = ["thermal", "structural", "electrical"]
_AUDIT_B = ["hydraulic", "acoustic", "radiological"]

_QUARTER_A = ["Q1", "Q3"]
_QUARTER_B = ["Q2", "Q4"]

_UNIT_A = ["ALPHA", "BRAVO", "CEDAR", "DELTA"]
_UNIT_B = ["ECHO", "FOXTROT", "GRANITE", "HALCYON"]


@dataclass(frozen=True)
class DrawPools:
    """Value pools for one corpus draw. A and B never intersect."""

    draw: str
    given: list[str]
    surname: list[str]
    color: list[str]
    sysroot: list[str]
    facility: list[str]
    role: list[str]
    place: list[str]
    idprefix: list[str]
    pathroot: list[str]
    host: list[str]
    audit: list[str]
    quarter: list[str]
    unit: list[str]
    year_lo: int
    year_hi: int
    num_lo: int
    num_hi: int


POOLS: dict[str, DrawPools] = {
    "A": DrawPools(
        draw="A", given=_GIVEN_A, surname=_SUR_A, color=_COLOR_A, sysroot=_SYSROOT_A,
        facility=_FACILITY_A, role=_ROLE_A, place=_PLACE_A, idprefix=_IDPREFIX_A,
        pathroot=_PATHROOT_A, host=_HOST_A, audit=_AUDIT_A, quarter=_QUARTER_A,
        unit=_UNIT_A, year_lo=2029, year_hi=2032, num_lo=10_000, num_hi=49_999,
    ),
    "B": DrawPools(
        draw="B", given=_GIVEN_B, surname=_SUR_B, color=_COLOR_B, sysroot=_SYSROOT_B,
        facility=_FACILITY_B, role=_ROLE_B, place=_PLACE_B, idprefix=_IDPREFIX_B,
        pathroot=_PATHROOT_B, host=_HOST_B, audit=_AUDIT_B, quarter=_QUARTER_B,
        unit=_UNIT_B, year_lo=2033, year_hi=2036, num_lo=50_000, num_hi=89_999,
    ),
}


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------

#: Answer-type hint appended to every question, in every condition.
#:
#: This constrains the *format* of the answer, never its content - each hint restates a
#: noun already present in its question. It exists because a 4B non-thinking model asked
#: "which facility does the person who maintains unit X operate out of?" reliably
#: completes hop 1 and answers with the person's name, which measures question ambiguity
#: rather than long-range composition. Because the hint is applied identically to NATIVE,
#: NOCTX and every compiled condition, it cannot bias a between-condition comparison.
ANSWER_HINTS = {
    "semantic": "a color",
    "identifier": "an assembly identifier",
    "number": "a number",
    "date": "a date",
    "hash": "a hexadecimal digest",
    "path": "a filesystem path",
    "url": "a URL",
    "proper_noun": "a person's name",
    # Measured on 4 draw-A docs at 1K: "a facility name" -> 1/4, this -> 3/4. The model
    # completes hop 1 and emits the bridging *person*; naming the answer type and ruling
    # out the bridge type fixes the format without supplying the answer. Qwen3-4B with
    # thinking enabled reaches 4/4 but spends 200-400 generated tokens per answer, which
    # is far too costly and variable for a harness that runs thousands of these.
    "composition": "a facility name, not a person's name",
}


@dataclass
class Fact:
    fact_id: str
    fact_class: str
    question: str
    answer: str
    answer_hint: str
    #: Same-class, similar-surface, wrong value living in a *different* chunk. Without
    #: this an exact-match score is unfalsifiable - the model can be right by guessing
    #: the format.
    distractor_answer: str
    #: Chunks that must be present for the fact to be answerable. Length 2 for
    #: `composition`, otherwise 1. Used as retrieval ground truth in Stage 6.
    support_chunks: list[int]
    distractor_chunks: list[int]


@dataclass
class Chunk:
    idx: int
    text: str
    n_tokens: int
    content_id: str


@dataclass
class Document:
    doc_id: str
    draw: str
    seed: int
    target_tokens: int
    chunk_tokens: int
    chunks: list[Chunk]
    facts: list[Fact]
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(c.text for c in self.chunks)

    @property
    def n_tokens(self) -> int:
        return sum(c.n_tokens for c in self.chunks)

    def to_json(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Filler
# --------------------------------------------------------------------------------------

_FILLER = [
    "The {sysroot} intake queue was drained on schedule and no anomalies were escalated.",
    "Routine calibration of the {sysroot} sensor bank completed without operator intervention.",
    "Staff at {facility} reported nominal humidity across the storage bays this cycle.",
    "The {quarter} inventory walkthrough for {facility} found no discrepancies worth recording.",
    "{person} filed the standing availability notice for the {sysroot} cluster.",
    "Transit through {place} remained unrestricted throughout the reporting window.",
    "A scheduled restart of the {sysroot} coordinator was deferred to the next maintenance slot.",
    "The {audit} checklist for {facility} was archived without amendment.",
    "Spare capacity at {facility} held steady and no reallocation was requested.",
    "Operators confirmed that the {sysroot} telemetry feed remained continuous.",
    "No outstanding tickets were carried over from the previous {quarter} review.",
    "The duty roster for {facility} was published in the usual format and circulated.",
    "Ambient load on the {sysroot} backplane stayed within the documented envelope.",
    "{person} acknowledged the routine notice regarding {facility} access hours.",
    "Housekeeping tasks for the {sysroot} archive ran to completion overnight.",
    "The {audit} sampling interval at {facility} was left unchanged this period.",
]


def _person(rng: random.Random, p: DrawPools) -> str:
    return f"{rng.choice(p.given)} {rng.choice(p.surname)}"


def _filler_sentence(rng: random.Random, p: DrawPools) -> str:
    return rng.choice(_FILLER).format(
        sysroot=rng.choice(p.sysroot),
        facility=rng.choice(p.facility),
        person=_person(rng, p),
        place=rng.choice(p.place),
        audit=rng.choice(p.audit),
        quarter=rng.choice(p.quarter),
    )


# --------------------------------------------------------------------------------------
# Fact construction
# --------------------------------------------------------------------------------------

def _sysname(rng: random.Random, p: DrawPools, serial: int) -> str:
    return f"{rng.choice(p.sysroot)}-{serial}"


def _confusable(name: str) -> str:
    """A near-miss variant of a system name, so distractors are genuinely confusable.

    `Mirren-4471` -> `Mirren-4472`. A distractor that shares no surface with the target
    is trivially ignored and would inflate every exact-match score.
    """
    root, _, serial = name.rpartition("-")
    return f"{root}-{int(serial) + 1}"


def _mk_value(rng: random.Random, p: DrawPools, fact_class: str, hash_hex: int) -> str:
    if fact_class == "semantic":
        return rng.choice(p.color)
    if fact_class == "identifier":
        return f"{rng.choice(p.idprefix)}-{rng.randint(1000, 9999)}-{rng.choice(p.idprefix)}"
    if fact_class == "number":
        return f"{rng.randint(p.num_lo, p.num_hi):,}"
    if fact_class == "date":
        return (
            f"{rng.randint(p.year_lo, p.year_hi)}-"
            f"{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        )
    if fact_class == "hash":
        return "".join(rng.choice("0123456789abcdef") for _ in range(hash_hex))
    if fact_class == "path":
        return f"{rng.choice(p.pathroot)}/spool/kx{rng.randint(10, 99)}.log"
    if fact_class == "url":
        return f"https://{rng.choice(p.host)}/rel/{rng.randint(1000, 9999)}"
    if fact_class == "proper_noun":
        return _person(rng, p)
    if fact_class == "composition":
        return rng.choice(p.facility)
    raise ValueError(f"unknown fact class {fact_class!r}")


def _render(fact_class: str, subject: str, value: str, rng: random.Random, p: DrawPools):
    """Return (sentences, question) for a fact.

    `sentences` is a list because `composition` deliberately splits across two chunks.
    """
    if fact_class == "semantic":
        who = subject
        return (
            [
                f"{who} bought a {value} bicycle after {who}'s car broke down "
                f"near {rng.choice(p.place)}."
            ],
            f"What color bicycle did {who} buy?",
        )
    if fact_class == "identifier":
        return (
            [f"The replacement assembly for {subject} is {value}."],
            f"What is the exact replacement assembly identifier for {subject}?",
        )
    if fact_class == "number":
        q = rng.choice(p.quarter)
        return (
            [f"The {q} reconciliation for {subject} totalled {value} credits."],
            f"What was the {q} reconciliation total for {subject}, in credits?",
        )
    if fact_class == "date":
        a = rng.choice(p.audit)
        return (
            [f"The {a} audit for {subject} closed on {value}."],
            f"On what date did the {a} audit for {subject} close?",
        )
    if fact_class == "hash":
        return (
            [f"The manifest digest for {subject} is {value}."],
            f"What is the manifest digest for {subject}?",
        )
    if fact_class == "path":
        return (
            [f"Diagnostics for {subject} are written to {value}."],
            f"To what path are diagnostics for {subject} written?",
        )
    if fact_class == "url":
        return (
            [f"The mirror for {subject} is published at {value}."],
            f"At what URL is the mirror for {subject} published?",
        )
    if fact_class == "proper_noun":
        return (
            [f"{value}, the {rng.choice(p.role)}, signed off on the {subject} handover."],
            f"Who signed off on the {subject} handover?",
        )
    if fact_class == "composition":
        # Two halves, placed in separated chunks by the caller. Neither half alone
        # answers the question.
        broker = _person(rng, p)
        return (
            [
                f"Unit {subject} is maintained by {broker}.",
                f"{broker} operates out of the {value} facility.",
            ],
            # Phrased to make both hops mandatory. An earlier wording
            # ("Which facility is responsible for maintaining unit X?") was answered
            # with the maintainer's *name* - the model completed hop 1 and stopped, so
            # the item measured question ambiguity rather than long-range composition.
            f"Which facility does the person who maintains unit {subject} operate out of?",
        )
    raise ValueError(fact_class)


# --------------------------------------------------------------------------------------
# Document generation
# --------------------------------------------------------------------------------------

TokenCounter = Callable[[str], int]


def approx_token_count(text: str) -> int:
    """Tokenizer-free approximation, for tests only. Never used to build real corpora."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def generate_document(
    *,
    draw: str,
    seed: int,
    doc_index: int,
    target_tokens: int,
    count_tokens: TokenCounter,
    chunk_tokens: int = 256,
    facts_per_class: int = 1,
    hash_hex: int = 16,
) -> Document:
    """Generate one deterministic document.

    Facts and their distractors are placed in *different* chunks; `composition` halves
    are placed at least two chunks apart so the model cannot solve them locally.
    """
    p = POOLS[draw]
    rng = random.Random(f"{draw}|{seed}|{doc_index}|{target_tokens}|{chunk_tokens}")

    n_chunks = max(2, round(target_tokens / chunk_tokens))

    # ---- build facts and decide placement -------------------------------------------
    # placement[c] is the list of sentences to inject into chunk c.
    placement: dict[int, list[str]] = {i: [] for i in range(n_chunks)}
    facts: list[Fact] = []
    serial = rng.randint(1000, 8000)

    for cls in FACT_CLASSES:
        for k in range(facts_per_class):
            serial += rng.randint(3, 40)
            subject = (
                rng.choice(p.unit) + f"-{serial}" if cls == "composition"
                else _person(rng, p) if cls == "semantic"
                else _sysname(rng, p, serial)
            )
            value = _mk_value(rng, p, cls, hash_hex)

            # Distractor: same template, near-miss subject, different value.
            d_subject = (
                subject.rsplit("-", 1)[0] + f"-{serial + 1}" if cls == "composition"
                else _person(rng, p) if cls == "semantic"
                else _confusable(subject)
            )
            d_value = _mk_value(rng, p, cls, hash_hex)
            guard = 0
            while d_value == value and guard < 20:
                d_value = _mk_value(rng, p, cls, hash_hex)
                guard += 1

            sents, question = _render(cls, subject, value, rng, p)
            d_sents, _ = _render(cls, d_subject, d_value, rng, p)

            if cls == "composition":
                # Halves must be separated; distractor halves go elsewhere again.
                lo = rng.randrange(0, max(1, n_chunks // 2))
                hi = rng.randrange(min(lo + 2, n_chunks - 1), n_chunks)
                if hi <= lo:
                    hi = min(lo + 1, n_chunks - 1)
                support = [lo, hi]
                d_lo = rng.randrange(0, n_chunks)
                d_hi = rng.randrange(0, n_chunks)
                dist = [d_lo, d_hi]
                placement[lo].append(sents[0])
                placement[hi].append(sents[1])
                placement[d_lo].append(d_sents[0])
                placement[d_hi].append(d_sents[1])
            else:
                c = rng.randrange(0, n_chunks)
                d_c = rng.randrange(0, n_chunks)
                guard = 0
                while d_c == c and n_chunks > 1 and guard < 20:
                    d_c = rng.randrange(0, n_chunks)
                    guard += 1
                support = [c]
                dist = [d_c]
                placement[c].append(sents[0])
                placement[d_c].append(d_sents[0])

            facts.append(
                Fact(
                    fact_id=f"{draw}-{doc_index}-{cls}-{k}",
                    fact_class=cls,
                    question=question,
                    answer=value,
                    answer_hint=ANSWER_HINTS[cls],
                    distractor_answer=d_value,
                    support_chunks=sorted(set(support)),
                    distractor_chunks=sorted(set(dist)),
                )
            )

    # ---- lay out chunks --------------------------------------------------------------
    chunks: list[Chunk] = []
    for i in range(n_chunks):
        header = f"### Section {i:03d}"
        body = list(placement[i])
        # Interleave the planted facts into filler at deterministic positions rather
        # than clumping them at the chunk head, which would make position trivially
        # informative.
        sentences: list[str] = []
        n_tok = count_tokens(header)
        pending = list(body)
        while n_tok < chunk_tokens:
            if pending and rng.random() < 0.35:
                s = pending.pop(0)
            else:
                s = _filler_sentence(rng, p)
            sentences.append(s)
            n_tok += count_tokens(" " + s)
        # Any facts that did not get interleaved before the budget ran out are appended
        # rather than dropped: a planted fact must never go missing.
        sentences.extend(pending)
        text = header + "\n" + " ".join(sentences)
        chunks.append(
            Chunk(
                idx=i,
                text=text,
                n_tokens=count_tokens(text),
                content_id=content_id(text),
            )
        )

    return Document(
        doc_id=f"{draw}-s{seed}-d{doc_index}-t{target_tokens}",
        draw=draw,
        seed=seed,
        target_tokens=target_tokens,
        chunk_tokens=chunk_tokens,
        chunks=chunks,
        facts=facts,
        meta={"n_chunks": n_chunks, "facts_per_class": facts_per_class, "hash_hex": hash_hex},
    )


def generate_corpus(
    *,
    draw: str,
    seed: int,
    lengths: Iterable[int],
    count_tokens: TokenCounter,
    n_docs: int = 8,
    chunk_tokens: int = 256,
    facts_per_class: int = 1,
) -> list[Document]:
    return [
        generate_document(
            draw=draw,
            seed=seed,
            doc_index=d,
            target_tokens=length,
            count_tokens=count_tokens,
            chunk_tokens=chunk_tokens,
            facts_per_class=facts_per_class,
        )
        for length in lengths
        for d in range(n_docs)
    ]


def save_corpus(docs: list[Document], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([d.to_json() for d in docs], fh, indent=1)


def load_corpus(path: str) -> list[Document]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    try:
        return _parse_corpus(raw)
    except TypeError as exc:
        raise TypeError(
            f"{path} was written by an older corpus schema ({exc}). Corpora are cheap to "
            "rebuild and draws A and B must be regenerated together - delete "
            f"{os.path.dirname(path) or '.'}/draw_*.json and re-run scripts/build_corpus.py "
            "for both draws."
        ) from exc


def _parse_corpus(raw: list[dict]) -> list[Document]:
    return [
        Document(
            doc_id=d["doc_id"], draw=d["draw"], seed=d["seed"],
            target_tokens=d["target_tokens"], chunk_tokens=d["chunk_tokens"],
            chunks=[Chunk(**c) for c in d["chunks"]],
            facts=[Fact(**f) for f in d["facts"]],
            meta=d.get("meta", {}),
        )
        for d in raw
    ]


def all_values(docs: Iterable[Document]) -> set[str]:
    """Every gold and distractor value in a corpus. Used to assert draw disjointness."""
    out: set[str] = set()
    for d in docs:
        for f in d.facts:
            out.add(f.answer)
            out.add(f.distractor_answer)
    return out
