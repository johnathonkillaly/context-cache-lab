#!/usr/bin/env python
"""Stage 2b evaluation: does learned compressed state beat equal-budget raw context?

The gate is `docs/EXPERIMENT.md` §7b, frozen before the compressor was written. The
primary comparator is **BUDGET**, not NOCTX:

    recovery(r) = ( LEARNED(r) − BUDGET(r) ) / ( NATIVE − BUDGET(r) )

Conditions per ratio: LEARNED (independent per-chunk compilation), LEARNED_JOINT (whole
document compiled as one chunk, which isolates the cost of *independence*), and BUDGET.
Plus the unreduced references NATIVE and NOCTX, and three controls at 4×:

* RANDOM      — memory states replaced by moment-matched noise
* WRONGPAGE   — states compiled from a *different* document
* SHUFFLE_PAGES — correct pages, permuted composition order

`SHUFFLE_PAGES` is a genuine control here, unlike Stage 2a's KV-order shuffle: because
pages are stored pre-RoPE and rotated at composition time, permuting them really does
change the position every page is encoded at.

Pages are compiled once per (document, ratio) and reused across that document's
questions — which is exactly the reuse the project is about.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from ccl.compress import budget_text  # noqa: E402
from ccl.compressor import MemoryExtractor  # noqa: E402
from ccl.corpus import Document, load_corpus  # noqa: E402
from ccl.metrics import aggregate, score  # noqa: E402
from ccl.provenance import stamp  # noqa: E402
from ccl.rope import cache_from_layers, compose_pages  # noqa: E402
from ccl.target import TargetModel  # noqa: E402

RATIOS = (2.0, 4.0, 8.0, 16.0)


def _free() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def _head_layers(tm: TargetModel, head_text: str):
    ids = tm.encode(head_text)
    n = ids.shape[1]
    pos = torch.arange(n, device=tm.device).unsqueeze(0)
    cache = tm.new_cache()
    tm.model(
        input_ids=ids, past_key_values=cache, position_ids=pos,
        cache_position=pos[0], use_cache=True,
    )
    return [(l.keys.clone(), l.values.clone()) for l in cache.layers], n


@torch.no_grad()
def _ask_composed(tm, head_layers, head_n, pages, tail_text, suffix_text, fact, max_new):
    composed, next_pos = compose_pages(tm.model, pages, start=head_n)
    merged = [
        (torch.cat([head_layers[i][0], composed[i][0]], dim=2),
         torch.cat([head_layers[i][1], composed[i][1]], dim=2))
        for i in range(len(composed))
    ]
    cache = cache_from_layers(tm.model, merged)
    feed = torch.cat([tm.encode(tail_text), tm.encode(suffix_text)], dim=1)
    n = feed.shape[1]
    pos = torch.arange(next_pos, next_pos + n, device=tm.device)
    cpos = torch.arange(cache.get_seq_length(), cache.get_seq_length() + n, device=tm.device)
    last_logits, cache = tm.prefill(feed, cache, position_ids=pos, cache_position=cpos)
    gold = tm.answer_logprob(fact.answer, prompt_last_logits=last_logits, cache=cache)
    dist = tm.answer_logprob(
        fact.distractor_answer, prompt_last_logits=last_logits, cache=cache
    )
    gen = tm.decode_from(last_logits, cache, max_new_tokens=max_new)
    row = score(fact, gen["text"])
    row.update(
        answer_logprob=gold["mean_logprob"],
        distractor_logprob=dist["mean_logprob"],
        rank_margin=gold["mean_logprob"] - dist["mean_logprob"],
        n_generated=gen["n_generated"],
        n_active_positions=int(cache.get_seq_length()),
    )
    del cache
    return row


@torch.no_grad()
def _ask_raw(tm, context, fact, max_new):
    ids = tm.encode(tm.build_prompt(fact.question, context, fact.answer_hint)) \
        if context is not None else \
        tm.encode(tm.build_prompt(fact.question, None, fact.answer_hint))
    last_logits, cache = tm.prefill(ids)
    gold = tm.answer_logprob(fact.answer, prompt_last_logits=last_logits, cache=cache)
    dist = tm.answer_logprob(
        fact.distractor_answer, prompt_last_logits=last_logits, cache=cache
    )
    gen = tm.decode_from(last_logits, cache, max_new_tokens=max_new)
    row = score(fact, gen["text"])
    row.update(
        answer_logprob=gold["mean_logprob"],
        distractor_logprob=dist["mean_logprob"],
        rank_margin=gold["mean_logprob"] - dist["mean_logprob"],
        n_generated=gen["n_generated"],
        n_active_positions=int(ids.shape[1]),
    )
    del cache
    return row


def _randomize_pages(pages, seed: int):
    """Moment-matched noise in place of the learned states."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for page in pages:
        layers = []
        for k, v in page:
            new = []
            for t in (k, v):
                mean = t.mean(dim=2, keepdim=True).float()
                std = t.std(dim=2, keepdim=True).float().clamp_min(1e-6)
                noise = torch.randn(t.shape, generator=g, dtype=torch.float32)
                new.append((noise.to(t.device) * std + mean).to(t.dtype))
            layers.append((new[0], new[1]))
        out.append(layers)
    return out


def run_doc(tm, extractor, doc: Document, other_doc: Document, args) -> list[dict]:
    rows: list[dict] = []
    fact0 = doc.facts[0]
    parts = tm.split_prompt4(fact0.question, doc.text, fact0.answer_hint)
    if parts is None:
        print(f"  SKIP {doc.doc_id}")
        return rows
    head, _body, tail, _sfx = parts
    head_layers, head_n = _head_layers(tm, head)
    n_doc_tokens = sum(tm.count_tokens(c.text) for c in doc.chunks)

    def emit(cond, ratio, pages, extra=None):
        for fact in doc.facts:
            p = tm.split_prompt4(fact.question, doc.text, fact.answer_hint)
            row = _ask_composed(
                tm, head_layers, head_n, pages, tail, p[3], fact, args.max_new_tokens
            )
            row.update(
                condition=cond, ratio=ratio, doc_id=doc.doc_id, draw=doc.draw,
                target_tokens=doc.target_tokens, n_chunks=len(doc.chunks),
                n_doc_tokens=n_doc_tokens,
                n_memory_positions=sum(pg[0][0].shape[2] for pg in pages),
                **(extra or {}),
            )
            rows.append(row)
        _free()

    chunk_tok = [tm.encode(c.text) for c in doc.chunks]

    for ratio in RATIOS:
        t0 = time.time()
        pages = extractor.compile_chunks(chunk_tok, ratio)
        compile_s = time.time() - t0
        emit("LEARNED", ratio, pages, {"compile_s": compile_s})

        # Whole document compiled as ONE chunk: isolates the cost of independence.
        joint = [extractor.extract(tm.encode(doc.text), ratio)]
        emit("LEARNED_JOINT", ratio, joint, {})
        del joint

        if ratio == args.control_ratio:
            emit("RANDOM", ratio, _randomize_pages(pages, seed=0), {})
            order = list(range(len(pages)))
            random.Random(0).shuffle(order)
            emit("SHUFFLE_PAGES", ratio, [pages[i] for i in order], {})
            if other_doc.doc_id == doc.doc_id:
                # Would silently reduce to LEARNED and look like an answer leak.
                print("  WARN: only one document - skipping WRONGPAGE control")
            else:
                wrong = extractor.compile_chunks(
                    [tm.encode(c.text) for c in other_doc.chunks], ratio
                )
                emit("WRONGPAGE", ratio, wrong, {})
                del wrong
        del pages
        _free()

    # Raw-text references
    for fact in doc.facts:
        for cond, ctx, ratio in (
            [("NATIVE", doc.text, 1.0), ("NOCTX", None, 1.0)]
            + [("BUDGET", budget_text([c.text for c in doc.chunks], r), r) for r in RATIOS]
        ):
            row = _ask_raw(tm, ctx, fact, args.max_new_tokens)
            row.update(
                condition=cond, ratio=ratio, doc_id=doc.doc_id, draw=doc.draw,
                target_tokens=doc.target_tokens, n_chunks=len(doc.chunks),
                n_doc_tokens=n_doc_tokens, n_memory_positions=None,
            )
            rows.append(row)
    _free()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--draw", choices=["A", "B"], default="A")
    ap.add_argument("--ckpt", default="results/raw/stage2b_extractor.pt")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--limit-docs", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--control-ratio", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--i-am-running-the-frozen-final-evaluation",
        action="store_true",
        help="required to touch held-out draw B",
    )
    args = ap.parse_args()

    if args.draw == "B" and not args.i_am_running_the_frozen_final_evaluation:
        raise SystemExit(
            "Draw B is held out for the single final evaluation against the frozen "
            "criteria in docs/EXPERIMENT.md §7b. Pass "
            "--i-am-running-the-frozen-final-evaluation only when that is what this is."
        )

    docs = [
        d for d in load_corpus(f"results/raw/corpus/draw_{args.draw}.json")
        if d.target_tokens == args.length
    ][: args.limit_docs]

    print(f"loading {args.model} ...", flush=True)
    tm = TargetModel(args.model)
    ck = torch.load(args.ckpt, map_location=tm.device, weights_only=False)
    extractor = MemoryExtractor(
        tm, layer_share=ck.get("layer_share", 1), n_sink=ck.get("n_sink", 8)
    )
    extractor.load_state_dict(ck["state_dict"])
    extractor.eval()
    print(f"loaded extractor from step {ck.get('step')} ({args.ckpt})")

    rows: list[dict] = []
    for i, doc in enumerate(docs):
        other = docs[(i + 1) % len(docs)]
        print(f"[{i + 1}/{len(docs)}] {doc.doc_id}", flush=True)
        rows.extend(run_doc(tm, extractor, doc, other, args))

    out = args.out or f"results/raw/stage2b_eval_draw{args.draw}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "stage": "2b-eval", "draw": args.draw, "model_id": args.model,
                "ckpt_step": ck.get("step"), "length": args.length,
                "n_documents": len(docs), "n_rows": len(rows),
                "ratios": list(RATIOS), "rows": rows, **stamp(),
            },
            fh, indent=1,
        )

    print("\n=== Stage 2b (clean_hit, pooled) ===")
    for cond, ratio in sorted({(r["condition"], r["ratio"]) for r in rows}):
        sub = [r for r in rows if r["condition"] == cond and r["ratio"] == ratio]
        a = aggregate(sub)["overall"]
        act = sum(r["n_active_positions"] for r in sub) / len(sub)
        print(
            f"{cond:16s} r={ratio:>4.0f}x  clean={a['clean_hit']:.3f} "
            f"margin={a.get('rank_margin', 0):+6.2f}  active={act:7.0f}"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
