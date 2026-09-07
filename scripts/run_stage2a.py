#!/usr/bin/env python
"""Stage 2a: training-free compression floor.

Before training a sidecar, find out what the frozen target does with a reduced cache
built by heuristics. This answers three things cheaply:

1. Does Qwen3-4B tolerate a sparsified KV cache at all, and up to what ratio?
2. How does that compare to spending the same token budget on **raw text** (`BUDGET`)?
   A learned compressor has to beat this line, not merely beat no-context.
3. Does keeping original absolute positions matter versus compacting them? This is the
   positional question Stage 3 turns on, asked here where it costs nothing to answer.

Efficiency note: the document is prefilled **once** per document into a master cache,
and every condition is then an `index_select` off that master. No condition re-runs the
document forward pass, so the sweep costs roughly one prefill per document rather than
one per condition.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from ccl.compress import (  # noqa: E402
    budget_text,
    keep_indices,
    randomize_cache,
    shuffle_cache_blocks,
    shuffle_document_text,
)
from ccl.corpus import Document, load_corpus  # noqa: E402
from ccl.metrics import aggregate, score  # noqa: E402
from ccl.provenance import stamp  # noqa: E402
from ccl.target import TargetModel  # noqa: E402

RATIOS = (2.0, 4.0, 8.0, 16.0)


def _free() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _clone_cache(tm: TargetModel, master: list[tuple[torch.Tensor, torch.Tensor]],
                 idx: torch.Tensor | None) -> DynamicCache:
    """Build a fresh cache from the master tensors, optionally selecting positions."""
    cache = tm.new_cache()
    for layer, (k, v) in enumerate(master):
        if idx is not None:
            k = k.index_select(2, idx)
            v = v.index_select(2, idx)
        cache.update(k.contiguous(), v.contiguous(), layer)
    return cache


def _ask(tm: TargetModel, cache: DynamicCache, suffix_ids: torch.Tensor, fact,
         query_pos: int, max_new_tokens: int) -> dict:
    """Run one question against an already-built context cache."""
    n = suffix_ids.shape[1]
    cache_pos = torch.arange(
        cache.get_seq_length(), cache.get_seq_length() + n, device=tm.device
    )
    pos = torch.arange(query_pos, query_pos + n, device=tm.device)
    last_logits, cache = tm.prefill(
        suffix_ids, cache, position_ids=pos, cache_position=cache_pos
    )
    gold = tm.answer_logprob(fact.answer, prompt_last_logits=last_logits, cache=cache)
    dist = tm.answer_logprob(
        fact.distractor_answer, prompt_last_logits=last_logits, cache=cache
    )
    gen = tm.decode_from(last_logits, cache, max_new_tokens=max_new_tokens)
    row = score(fact, gen["text"])
    row.update(
        answer_logprob=gold["mean_logprob"],
        distractor_logprob=dist["mean_logprob"],
        rank_margin=gold["mean_logprob"] - dist["mean_logprob"],
        n_generated=gen["n_generated"],
    )
    return row


def run_doc(tm: TargetModel, doc: Document, max_new_tokens: int) -> list[dict]:
    parts = tm.split_prompt4(doc.facts[0].question, doc.text, doc.facts[0].answer_hint)
    if parts is None:
        print(f"  SKIP {doc.doc_id}: prompt does not partition on token boundaries")
        return []
    head, body, tail, _ = parts
    h, b, t = (tm.encode(x) for x in (head, body, tail))

    # One document prefill; every condition below is an index_select off this.
    _, master_cache = tm.prefill(h)
    _, master_cache = tm.prefill(b, master_cache)
    _, master_cache = tm.prefill(t, master_cache)
    doc_start, doc_end = int(h.shape[1]), int(h.shape[1] + b.shape[1])
    n_doc = doc_end - doc_start
    full_len = master_cache.get_seq_length()
    master = [(lay.keys.clone(), lay.values.clone()) for lay in master_cache.layers]
    del master_cache
    _free()

    rows: list[dict] = []

    def emit(cond: str, ratio: float, strategy: str, posmode: str,
             idx: list[int] | None, n_active: int, *, mutate=None) -> None:
        for fact in doc.facts:
            s = tm.split_prompt4(fact.question, doc.text, fact.answer_hint)
            suffix_ids = tm.encode(s[3])
            sel = torch.tensor(idx, device=tm.device) if idx is not None else None
            cache = _clone_cache(tm, master, sel)
            if mutate is not None:
                mutate(cache)
            # `original` keeps the query at the position it would have had with the
            # full document; `compact` re-dates it to the shrunken length.
            query_pos = full_len if posmode == "original" else cache.get_seq_length()
            row = _ask(tm, cache, suffix_ids, fact, query_pos, max_new_tokens)
            row.update(
                condition=cond, ratio=ratio, strategy=strategy, position_mode=posmode,
                doc_id=doc.doc_id, draw=doc.draw, target_tokens=doc.target_tokens,
                n_chunks=len(doc.chunks), n_doc_tokens=n_doc,
                n_active_positions=n_active,
                active_compression=n_doc / max(1, n_active - (full_len - n_doc)),
            )
            rows.append(row)
            del cache
        _free()

    keep_all = list(range(full_len))

    # -- reference: the full cache, reconstructed the same way as every other condition
    emit("NATIVE_CACHED", 1.0, "none", "original", keep_all, full_len)

    # -- controls that do not reduce anything, so any drop is attributable to the
    #    manipulation rather than to a smaller budget
    emit("RANDOM", 1.0, "random_kv", "original", keep_all, full_len,
         mutate=lambda c: randomize_cache(c, doc_start, doc_end, seed=0))
    # Expected to be an exact no-op - see compress.shuffle_cache_blocks. Run as a
    # positive check that position lives in the rotated keys, not in tensor order.
    chunk_span = max(1, n_doc // max(1, len(doc.chunks)))
    emit("SHUFFLE_KV_ORDER", 1.0, "shuffle_blocks", "original", keep_all, full_len,
         mutate=lambda c: shuffle_cache_blocks(c, doc_start, doc_end, chunk_span, seed=0))

    # -- the compression sweep
    for ratio in RATIOS:
        for strategy in ("stride", "sink"):
            kept = keep_indices(n_doc, ratio, strategy)
            idx = list(range(doc_start)) + [doc_start + i for i in kept] + \
                list(range(doc_end, full_len))
            emit(f"KV_{strategy.upper()}", ratio, strategy, "original", idx, len(idx))
        # positional ablation: same retained keys, query re-dated to the compact length
        kept = keep_indices(n_doc, ratio, "stride")
        idx = list(range(doc_start)) + [doc_start + i for i in kept] + \
            list(range(doc_end, full_len))
        emit("KV_STRIDE_COMPACT", ratio, "stride", "compact", idx, len(idx))

    del master
    _free()

    # -- conditions that need their own prefill because the text itself changes.
    chunk_texts = [c.text for c in doc.chunks]
    variants = [("BUDGET", r, budget_text(chunk_texts, r), "raw_uniform_chunks")
                for r in RATIOS]
    # Genuinely reorders chunks, so positions change - unlike SHUFFLE_KV_ORDER.
    variants.append(
        ("SHUFFLE_TEXT", 1.0, shuffle_document_text(chunk_texts, seed=0), "raw_shuffled")
    )
    for cond_name, ratio, text, strat in variants:
        for fact in doc.facts:
            ids = tm.encode(tm.build_prompt(fact.question, text, fact.answer_hint))
            last_logits, cache = tm.prefill(ids)
            gold = tm.answer_logprob(
                fact.answer, prompt_last_logits=last_logits, cache=cache
            )
            dist = tm.answer_logprob(
                fact.distractor_answer, prompt_last_logits=last_logits, cache=cache
            )
            gen = tm.decode_from(last_logits, cache, max_new_tokens=max_new_tokens)
            row = score(fact, gen["text"])
            row.update(
                condition=cond_name, ratio=ratio, strategy=strat,
                position_mode="native", doc_id=doc.doc_id, draw=doc.draw,
                target_tokens=doc.target_tokens, n_chunks=len(doc.chunks),
                n_doc_tokens=n_doc, n_active_positions=int(ids.shape[1]),
                active_compression=n_doc / max(1, tm.count_tokens(text)),
                answer_logprob=gold["mean_logprob"],
                distractor_logprob=dist["mean_logprob"],
                rank_margin=gold["mean_logprob"] - dist["mean_logprob"],
                n_generated=gen["n_generated"],
            )
            rows.append(row)
            del cache
        _free()

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", choices=["A", "B"], default="A")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--corpus-dir", default="results/raw/corpus")
    ap.add_argument("--lengths", default="4096,16384")
    ap.add_argument("--limit-docs", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.draw == "B":
        raise SystemExit(
            "Draw B is held out for the single final evaluation (docs/EXPERIMENT.md "
            "sections 3 and 7). Refusing to run an exploratory sweep against it."
        )

    docs = load_corpus(os.path.join(args.corpus_dir, f"draw_{args.draw}.json"))
    keep = {int(x) for x in args.lengths.split(",")}
    docs = [d for d in docs if d.target_tokens in keep]
    seen: dict[int, int] = {}
    picked = []
    for d in docs:
        c = seen.get(d.target_tokens, 0)
        if c < args.limit_docs:
            picked.append(d)
            seen[d.target_tokens] = c + 1
    docs = picked

    print(f"loading {args.model} ...", flush=True)
    tm = TargetModel(args.model)

    rows: list[dict] = []
    for i, doc in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] {doc.doc_id} ({doc.n_tokens} tok)", flush=True)
        rows.extend(run_doc(tm, doc, args.max_new_tokens))

    out = args.out or f"results/raw/stage2a_trainfree_draw{args.draw}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "stage": "2a",
                "draw": args.draw,
                "model_id": args.model,
                "ratios": list(RATIOS),
                "n_documents": len(docs),
                "n_rows": len(rows),
                "rows": rows,
                **stamp(),
            },
            fh,
            indent=1,
        )

    print("\n=== Stage 2a (clean_hit, pooled) ===")
    keys = sorted({(r["condition"], r["ratio"]) for r in rows}, key=lambda x: (x[0], x[1]))
    for cond, ratio in keys:
        sub = [r for r in rows if r["condition"] == cond and r["ratio"] == ratio]
        agg = aggregate(sub)["overall"]
        active = sum(r["n_active_positions"] for r in sub) / len(sub)
        print(
            f"{cond:20s} r={ratio:>4.0f}x  clean={agg['clean_hit']:.3f} "
            f"margin={agg.get('rank_margin', 0):+6.2f}  active={active:7.0f}"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
