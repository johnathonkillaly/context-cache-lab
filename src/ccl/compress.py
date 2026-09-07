"""Stage 2a: training-free KV compression, and the controls that go with it.

Purpose: establish the **floor** before training anything. If the frozen target can
already use a 4x-reduced cache built by a one-line heuristic, a learned compressor has
to beat that, not merely beat no-context. And if the frozen target collapses at 2x even
with the gentlest possible reduction, that is important to know before writing a
training loop.

Everything here operates on a `DynamicCache` that already holds the document's KV, so
the keys are **post-RoPE**. That constrains what is legitimate:

* Dropping positions is fine - each retained key keeps the RoPE phase it was encoded
  with, so the retained subset is exactly what full attention would have seen for those
  positions.
* Averaging or otherwise mixing keys across positions is NOT fine here, because two
  post-RoPE keys at different positions have been rotated by different angles and their
  mean is not the key of anything. Doing that properly needs pre-RoPE keys, which is
  Stage 2's job, not this module's.

That asymmetry is the whole reason C2KV stores KV pre-RoPE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache

#: How to renumber the query's positions once document positions have been removed.
#:
#: `original` keeps every retained key at its original absolute position, so the query
#: sits at the original document end. Relative distances between surviving tokens are
#: preserved exactly, and there are simply gaps.
#:
#: `compact` renumbers survivors to 0..m-1, so the query sits at m. Distances shrink.
#: This is what a naive "just concatenate the compiled states" implementation does, and
#: it is only correct if the keys are re-rotated - which post-RoPE keys cannot be. It is
#: included precisely so the cost of getting this wrong is measured rather than assumed.
POSITION_MODES = ("original", "compact")


@dataclass
class CompressionPlan:
    """Which document positions survive, and what it cost."""

    strategy: str
    ratio: float
    kept: list[int]
    n_original: int

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def achieved_ratio(self) -> float:
        return self.n_original / max(1, self.n_kept)


def keep_indices(n: int, ratio: float, strategy: str = "stride", n_sink: int = 4) -> list[int]:
    """Choose which of `n` document positions to retain at approximately `1/ratio`.

    Strategies:
      * `stride`  - uniformly spaced. The obvious baseline.
      * `sink`    - the first `n_sink` positions plus a uniform stride over the rest.
        Attention-sink work (StreamingLLM, and EPIC in the KV-reuse line) finds the
        earliest positions absorb a large share of attention mass regardless of content,
        so dropping them is disproportionately damaging.
      * `head`    - a contiguous prefix. Deliberately bad; it is the truncation strawman
        and exists to show that *where* you spend the budget matters.
      * `tail`    - a contiguous suffix, ditto.
    """
    if n <= 0:
        return []
    target = max(1, round(n / ratio))
    if strategy == "stride":
        return _uniform(n, target)
    if strategy == "sink":
        sink = list(range(min(n_sink, n)))
        rest = [i for i in _uniform(n, max(1, target - len(sink))) if i >= len(sink)]
        return sorted(set(sink + rest))
    if strategy == "head":
        return list(range(min(target, n)))
    if strategy == "tail":
        return list(range(max(0, n - target), n))
    raise ValueError(f"unknown strategy {strategy!r}")


def _uniform(n: int, target: int) -> list[int]:
    if target >= n:
        return list(range(n))
    step = n / target
    return sorted({min(n - 1, int(i * step)) for i in range(target)})


def compress_cache(
    cache: DynamicCache, doc_start: int, doc_end: int, kept: list[int]
) -> list[int]:
    """Retain only `kept` (document-relative) positions inside [doc_start, doc_end).

    Everything outside the document span - the chat/system head and the closing tags -
    is preserved untouched. Mutates `cache` in place and returns the surviving absolute
    positions, which the caller needs in order to place the query correctly.
    """
    n_total = cache.get_seq_length()
    head = list(range(doc_start))
    doc = [doc_start + i for i in kept]
    tail = list(range(doc_end, n_total))
    survivors = head + doc + tail
    idx = torch.tensor(survivors, device=cache.layers[0].keys.device)
    for layer in cache.layers:
        layer.keys = layer.keys.index_select(2, idx).contiguous()
        layer.values = layer.values.index_select(2, idx).contiguous()
    return survivors


def randomize_cache(cache: DynamicCache, doc_start: int, doc_end: int, seed: int = 0) -> None:
    """RANDOM control: replace document KV with noise matched per layer and per head.

    Matching the first two moments matters. Zero or standard-normal tensors would be
    off-distribution enough that the model could plausibly *ignore* them, which would
    make the control easier to beat than it should be. Matched noise carries no
    information but looks statistically like real cache content.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            span = tensor[:, :, doc_start:doc_end, :]
            mean = span.mean(dim=2, keepdim=True).float()
            std = span.std(dim=2, keepdim=True).float().clamp_min(1e-6)
            noise = torch.randn(span.shape, generator=g, dtype=torch.float32)
            tensor[:, :, doc_start:doc_end, :] = (
                (noise.to(mean.device) * std + mean).to(tensor.dtype)
            )


def shuffle_cache_blocks(
    cache: DynamicCache, doc_start: int, doc_end: int, block: int, seed: int = 0
) -> None:
    """Permute whole blocks of document KV in the cache tensors.

    **This is expected to be an exact no-op, and that is why it is worth running.**
    Attention is permutation-invariant over keys, and a post-RoPE key already carries
    its own position in its rotation - so reordering the tensors changes nothing that
    the query can observe. Measured: identical to the uncompressed condition to three
    decimal places, including rank margins.

    It therefore is *not* a test of whether chunk order matters. It is a positive check
    that position lives in the rotated keys rather than in tensor order, which is
    precisely the property that forces C2KV to store KV **pre**-RoPE: you cannot
    re-place an independently encoded chunk without re-rotating it. The meaningful
    shuffle control belongs in Stage 3, where positions are re-applied at composition
    time; `shuffle_document_text` below is its training-free stand-in.
    """
    n = doc_end - doc_start
    n_blocks = max(1, n // block)
    order = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(seed))
    idx: list[int] = []
    for b in order.tolist():
        lo = doc_start + b * block
        hi = doc_start + (b + 1) * block if b < n_blocks - 1 else doc_end
        idx.extend(range(lo, hi))
    full = list(range(doc_start)) + idx + list(range(doc_end, cache.get_seq_length()))
    sel = torch.tensor(full, device=cache.layers[0].keys.device)
    for layer in cache.layers:
        layer.keys = layer.keys.index_select(2, sel).contiguous()
        layer.values = layer.values.index_select(2, sel).contiguous()


def shuffle_document_text(chunks: list[str], seed: int = 0) -> str:
    """Reorder the document's chunks in the raw text, then re-prefill.

    Unlike `shuffle_cache_blocks` this genuinely changes the position each chunk is
    encoded at, so it does measure whether chunk order matters. It is the training-free
    stand-in for Stage 3's shuffled-composition control, and it establishes a reference:
    if reordering raw chunks already costs nothing on this corpus, then a compiled
    condition surviving a shuffle proves less than it appears to.
    """
    order = torch.randperm(len(chunks), generator=torch.Generator().manual_seed(seed))
    return "\n\n".join(chunks[i] for i in order.tolist())


def budget_text(chunks: list[str], ratio: float) -> str:
    """BUDGET control: raw text subsampled to ~1/ratio of the chunks, uniformly spread.

    This is the comparison the KV-reuse literature most often omits and the one that
    decides whether compression is doing useful work: a compiled state that beats
    no-context but loses to the same number of *raw* tokens has demonstrated nothing.

    Uniform spread rather than a prefix, so the control is a fair use of the budget
    rather than a strawman.
    """
    keep = _uniform(len(chunks), max(1, round(len(chunks) / ratio)))
    return "\n\n".join(chunks[i] for i in keep)
