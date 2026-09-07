"""Stage 2b training: compression-concatenation co-training.

The essential property, from C²KV: chunks are **extracted independently** but
**supervised only after concatenation**. Gradients flow from the answer loss back through
the composition into every chunk's extractor, so the sidecar cannot settle on states that
are individually informative but do not compose. Independence at extraction is what makes
Stage 3 meaningful; supervision after composition is what makes it work.

Loss is on **answer tokens only** — no reconstruction / autoencoding term, and no
next-token objective over the corpus (Cartridges reports the latter is not competitive
with in-context learning).

The compressor never sees a question. `extract()` takes a chunk and nothing else, so
question-specific shortcuts are unrepresentable rather than merely discouraged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .compressor import MemoryExtractor
from .corpus import Document, Fact
from .rope import cache_from_layers, compose_pages


@dataclass
class Example:
    doc: Document
    fact: Fact
    chunk_ids: list[int]


def build_examples(
    docs: list[Document], rng: random.Random, min_chunks: int = 2, max_chunks: int = 6
) -> list[Example]:
    """One example per fact: the fact's support chunks plus random filler chunks.

    Support chunks are always included - training a compressor on contexts that do not
    contain the answer would teach it to hallucinate rather than to compress. Filler
    chunks are sampled so the model must encode content it will not be asked about,
    which is the realistic case.
    """
    out: list[Example] = []
    for doc in docs:
        n = len(doc.chunks)
        for fact in doc.facts:
            need = sorted(set(fact.support_chunks))
            k = rng.randint(max(min_chunks, len(need)), max(min_chunks, min(max_chunks, n)))
            pool = [i for i in range(n) if i not in need]
            rng.shuffle(pool)
            chosen = sorted(set(need + pool[: max(0, k - len(need))]))
            out.append(Example(doc=doc, fact=fact, chunk_ids=chosen))
    return out


def compose_context_cache(
    tm,
    extractor: MemoryExtractor | None,
    doc: Document,
    chunk_ids: list[int],
    ratio: float,
    head_text: str,
    pages=None,
):
    """Build [head tokens][compiled pages][...] as a cache, returning (cache, next_pos).

    The head (chat scaffolding plus the opening `<document>` tag) is prefilled as ordinary
    tokens under `no_grad`; only the pages carry gradient. Pages land at contiguous
    positions immediately after the head, and each page's keys are rotated to where it
    actually lands - see `ccl.rope.compose_pages`.
    """
    head_ids = tm.encode(head_text)
    with torch.no_grad():
        h_n = head_ids.shape[1]
        pos = torch.arange(h_n, device=tm.device).unsqueeze(0)
        head_cache = tm.new_cache()
        tm.model(
            input_ids=head_ids,
            past_key_values=head_cache,
            position_ids=pos,
            cache_position=pos[0],
            use_cache=True,
        )
        head_layers = [(l.keys.detach(), l.values.detach()) for l in head_cache.layers]

    if pages is None:
        chunk_tok = [tm.encode(doc.chunks[i].text) for i in chunk_ids]
        pages = extractor.compile_chunks(chunk_tok, ratio)

    composed, next_pos = compose_pages(tm.model, pages, start=h_n)
    merged = [
        (torch.cat([head_layers[i][0], composed[i][0]], dim=2),
         torch.cat([head_layers[i][1], composed[i][1]], dim=2))
        for i in range(len(composed))
    ]
    return cache_from_layers(tm.model, merged), next_pos


def answer_loss(tm, cache, next_pos: int, tail_text: str, suffix_text: str, answer: str):
    """Cross-entropy over the answer tokens, conditioned on the composed context.

    Feeds [closing tags][question suffix][answer] in one pass so the whole thing is a
    single forward/backward through the frozen target.
    """
    tail_ids = tm.encode(tail_text)
    suffix_ids = tm.encode(suffix_text)
    ans_ids = tm.encode(answer)
    feed = torch.cat([tail_ids, suffix_ids, ans_ids], dim=1)
    n = feed.shape[1]
    n_ans = ans_ids.shape[1]

    pos = torch.arange(next_pos, next_pos + n, device=tm.device)
    cache_pos = torch.arange(
        cache.get_seq_length(), cache.get_seq_length() + n, device=tm.device
    )
    out = tm.model(
        input_ids=feed,
        past_key_values=cache,
        position_ids=pos.unsqueeze(0),
        cache_position=cache_pos,
        use_cache=True,
    )
    # logits[i] predicts feed[i+1]; the answer occupies the final n_ans positions.
    logits = out.logits[0, n - n_ans - 1 : n - 1, :]
    loss = torch.nn.functional.cross_entropy(logits.float(), ans_ids[0])
    return loss, n_ans


def example_texts(tm, doc: Document, fact: Fact, chunk_ids: list[int]):
    """The three text pieces around the compiled pages, from the real prompt builder.

    Derived by splitting an actual prompt rather than reassembled by hand, so training
    and evaluation cannot drift apart in whitespace or template details.
    """
    sub = "\n\n".join(doc.chunks[i].text for i in chunk_ids)
    parts = tm.split_prompt4(fact.question, sub, fact.answer_hint)
    if parts is None:
        return None
    head, _body, tail, suffix = parts
    return head, tail, suffix


def train_step(
    tm,
    extractor: MemoryExtractor,
    ex: Example,
    ratio: float,
) -> tuple[torch.Tensor, int]:
    texts = example_texts(tm, ex.doc, ex.fact, ex.chunk_ids)
    if texts is None:
        return None, 0
    head, tail, suffix = texts
    cache, next_pos = compose_context_cache(
        tm, extractor, ex.doc, ex.chunk_ids, ratio, head
    )
    return answer_loss(tm, cache, next_pos, tail, suffix, ex.fact.answer)


@torch.no_grad()
def eval_rank_margin(tm, extractor: MemoryExtractor, ex: Example, ratio: float) -> float | None:
    """logprob(gold) - logprob(distractor) under the compiled context.

    Lower variance than generation accuracy, so it moves visibly during a short smoke
    run when accuracy is still at floor.
    """
    texts = example_texts(tm, ex.doc, ex.fact, ex.chunk_ids)
    if texts is None:
        return None
    head, tail, suffix = texts
    # Compile once and score both candidates against the same pages: compilation is the
    # expensive half, and re-running it per candidate would also let the two answers be
    # scored against numerically different states.
    chunk_tok = [tm.encode(ex.doc.chunks[i].text) for i in ex.chunk_ids]
    pages = extractor.compile_chunks(chunk_tok, ratio)
    margins = []
    for answer in (ex.fact.answer, ex.fact.distractor_answer):
        cache, next_pos = compose_context_cache(
            tm, extractor, ex.doc, ex.chunk_ids, ratio, head, pages=pages
        )
        loss, _ = answer_loss(tm, cache, next_pos, tail, suffix, answer)
        margins.append(-float(loss))  # mean logprob of that answer
        del cache
    return margins[0] - margins[1]


def assert_target_frozen(tm, extractor: MemoryExtractor) -> None:
    """No gradient may ever reach the target. Checked after backward, not assumed."""
    leaked = [
        n for n, p in tm.model.named_parameters() if p.grad is not None or p.requires_grad
    ]
    if leaked:
        raise RuntimeError(
            f"target model is not frozen: {len(leaked)} params have grad "
            f"(e.g. {leaked[:3]}). Stage 2b requires a frozen target."
        )
