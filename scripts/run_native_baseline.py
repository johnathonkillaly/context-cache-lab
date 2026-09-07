#!/usr/bin/env python
"""Stage 1c: native baseline (NATIVE and NOCTX) plus native prefill timing.

This establishes the two reference points every later stage is measured between:
the upper bound (full raw context) and the floor (no context at all).

Stage 1 gate (docs/EXPERIMENT.md section 6): NATIVE must clearly beat NOCTX on every
fact class, and NOCTX must sit near the floor on the exact classes. If NOCTX scores well
on exact classes the corpus is contaminated and must be regenerated before anything else
is built.

Quality runs reuse the document's KV prefix across that document's questions. Exact
prefix reuse is mathematically identical to a fresh prefill, so it changes speed only.
Timing runs deliberately do NOT reuse, so the reported native prefill cost is honest.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from ccl.corpus import Document, load_corpus  # noqa: E402
from ccl.metrics import aggregate, score  # noqa: E402
from ccl.provenance import stamp  # noqa: E402
from ccl.target import TargetModel, crop_to  # noqa: E402
from ccl.timing import measure, sync  # noqa: E402


def _free() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_native_for_doc(tm: TargetModel, doc: Document, max_new_tokens: int) -> list[dict]:
    """NATIVE condition for one document, reusing the document KV prefix."""
    rows: list[dict] = []
    split = tm.split_prompt(doc.facts[0].question, doc.text, doc.facts[0].answer_hint)
    if split is None:
        # BPE did not tokenize prefix+suffix as the concatenation; fall back to full
        # prefill per question rather than silently reusing a misaligned cache.
        for fact in doc.facts:
            rows.append(_native_no_reuse(tm, doc, fact, max_new_tokens))
        return rows

    prefix_text, _ = split
    prefix_ids = tm.encode(prefix_text)
    _, cache = tm.prefill(prefix_ids)
    prefix_len = cache.get_seq_length()

    for fact in doc.facts:
        s = tm.split_prompt(fact.question, doc.text, fact.answer_hint)
        assert s is not None and s[0] == prefix_text, "prefix drifted between questions"
        suffix_ids = tm.encode(s[1])
        last_logits, cache = tm.prefill(suffix_ids, cache)
        prompt_len = cache.get_seq_length()

        gold = tm.answer_logprob(fact.answer, prompt_last_logits=last_logits, cache=cache)
        dist = tm.answer_logprob(
            fact.distractor_answer, prompt_last_logits=last_logits, cache=cache
        )
        gen = tm.decode_from(last_logits, cache, max_new_tokens=max_new_tokens)
        crop_to(cache, prefix_len)

        row = score(fact, gen["text"])
        row.update(
            condition="NATIVE",
            doc_id=doc.doc_id,
            draw=doc.draw,
            target_tokens=doc.target_tokens,
            n_chunks=len(doc.chunks),
            n_prompt_tokens=prompt_len,
            n_active_positions=prompt_len,
            answer_logprob=gold["mean_logprob"],
            distractor_logprob=dist["mean_logprob"],
            rank_margin=gold["mean_logprob"] - dist["mean_logprob"],
            n_generated=gen["n_generated"],
        )
        rows.append(row)

    del cache
    _free()
    return rows


def _native_no_reuse(tm: TargetModel, doc: Document, fact, max_new_tokens: int) -> dict:
    ids = tm.encode(tm.build_prompt(fact.question, doc.text, fact.answer_hint))
    last_logits, cache = tm.prefill(ids)
    gold = tm.answer_logprob(fact.answer, prompt_last_logits=last_logits, cache=cache)
    dist = tm.answer_logprob(
        fact.distractor_answer, prompt_last_logits=last_logits, cache=cache
    )
    gen = tm.decode_from(last_logits, cache, max_new_tokens=max_new_tokens)
    row = score(fact, gen["text"])
    row.update(
        condition="NATIVE",
        doc_id=doc.doc_id,
        draw=doc.draw,
        target_tokens=doc.target_tokens,
        n_chunks=len(doc.chunks),
        n_prompt_tokens=int(ids.shape[1]),
        n_active_positions=int(ids.shape[1]),
        answer_logprob=gold["mean_logprob"],
        distractor_logprob=dist["mean_logprob"],
        rank_margin=gold["mean_logprob"] - dist["mean_logprob"],
        n_generated=gen["n_generated"],
        prefix_reuse=False,
    )
    del cache
    return row


def run_noctx_for_doc(tm: TargetModel, doc: Document, max_new_tokens: int) -> list[dict]:
    """NOCTX control: question only. Anything above floor here is parametric leakage."""
    rows: list[dict] = []
    for fact in doc.facts:
        ids = tm.encode(tm.build_prompt(fact.question, None, fact.answer_hint))
        last_logits, cache = tm.prefill(ids)
        gold = tm.answer_logprob(fact.answer, prompt_last_logits=last_logits, cache=cache)
        dist = tm.answer_logprob(
            fact.distractor_answer, prompt_last_logits=last_logits, cache=cache
        )
        gen = tm.decode_from(last_logits, cache, max_new_tokens=max_new_tokens)
        row = score(fact, gen["text"])
        row.update(
            condition="NOCTX",
            doc_id=doc.doc_id,
            draw=doc.draw,
            target_tokens=doc.target_tokens,
            n_chunks=len(doc.chunks),
            n_prompt_tokens=int(ids.shape[1]),
            n_active_positions=int(ids.shape[1]),
            answer_logprob=gold["mean_logprob"],
            distractor_logprob=dist["mean_logprob"],
            rank_margin=gold["mean_logprob"] - dist["mean_logprob"],
            n_generated=gen["n_generated"],
        )
        rows.append(row)
        del cache
    _free()
    return rows


def timing_sweep(tm: TargetModel, docs: list[Document], trials: int, warmup: int) -> list[dict]:
    """Cold native prefill + TTFT per context length. No prefix reuse anywhere here."""
    out: list[dict] = []
    by_len: dict[int, Document] = {}
    for d in docs:
        by_len.setdefault(d.target_tokens, d)

    # Global warm-up at the largest shape before any measurement. MPS compiles kernels
    # per shape on first use, and the first length measured otherwise absorbs that cost:
    # an earlier run showed the 1K prefill samples falling monotonically 2.25s -> 1.50s
    # across five trials, which made 1K look *slower* than 2K.
    if by_len:
        biggest = by_len[max(by_len)]
        warm_ids = tm.encode(
            tm.build_prompt(
                biggest.facts[0].question, biggest.text, biggest.facts[0].answer_hint
            )
        )
        for _ in range(2):
            _, c = tm.prefill(warm_ids)
            del c
        _free()

    for length, doc in sorted(by_len.items()):
        fact = doc.facts[0]
        ids = tm.encode(tm.build_prompt(fact.question, doc.text, fact.answer_hint))
        n = int(ids.shape[1])

        def _prefill_only() -> None:
            _, cache = tm.prefill(ids)
            del cache

        _, prefill_stats = measure(_prefill_only, trials=trials, warmup=warmup)

        def _ttft() -> float:
            g = tm.generate_greedy(ids, max_new_tokens=1)
            return g["ttft_s"]

        _, ttft_stats = measure(_ttft, trials=trials, warmup=warmup)

        sync()
        mem = tm.memory_report()
        out.append(
            {
                "target_tokens": length,
                "n_prompt_tokens": n,
                "prefill_median_s": prefill_stats["median_s"],
                "prefill_stats": prefill_stats,
                "ttft_median_s": ttft_stats["median_s"],
                "ttft_stats": ttft_stats,
                "prefill_tokens_per_s": n / prefill_stats["median_s"],
                "kv_bytes": tm.geometry.kv_bytes_per_token * n,
                "memory": mem,
            }
        )
        print(
            f"  [{length:>6}] n={n:>6} prefill={prefill_stats['median_s']:.3f}s "
            f"({n / prefill_stats['median_s']:.0f} tok/s) ttft={ttft_stats['median_s']:.3f}s",
            flush=True,
        )
        _free()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", choices=["A", "B"], default="A")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--corpus-dir", default="results/raw/corpus")
    ap.add_argument("--lengths", default=None, help="comma list; default all in corpus")
    ap.add_argument("--limit-docs", type=int, default=None, help="docs per length")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument(
        "--timing-only",
        action="store_true",
        help="re-measure latency without redoing the quality sweep",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    docs = load_corpus(os.path.join(args.corpus_dir, f"draw_{args.draw}.json"))
    if args.lengths:
        keep = {int(x) for x in args.lengths.split(",")}
        docs = [d for d in docs if d.target_tokens in keep]
    if args.limit_docs:
        seen: dict[int, int] = {}
        kept = []
        for d in docs:
            c = seen.get(d.target_tokens, 0)
            if c < args.limit_docs:
                kept.append(d)
                seen[d.target_tokens] = c + 1
        docs = kept

    print(f"loading {args.model} ...", flush=True)
    tm = TargetModel(args.model)
    print(f"device={tm.device} params={sum(p.numel() for p in tm.model.parameters()):,}")

    rows: list[dict] = []
    if not args.timing_only:
        for i, doc in enumerate(docs, 1):
            print(f"[{i}/{len(docs)}] {doc.doc_id} ({doc.n_tokens} tok)", flush=True)
            rows.extend(run_native_for_doc(tm, doc, args.max_new_tokens))
            rows.extend(run_noctx_for_doc(tm, doc, args.max_new_tokens))

    timings: list[dict] = []
    if not args.skip_timing:
        print("timing sweep (cold native prefill, no reuse):", flush=True)
        timings = timing_sweep(tm, docs, args.trials, args.warmup)

    # Aggregate per (condition, length)
    summary: dict = {}
    for cond in ("NATIVE", "NOCTX"):
        summary[cond] = {}
        for length in sorted({d.target_tokens for d in docs}):
            sub = [
                r for r in rows if r["condition"] == cond and r["target_tokens"] == length
            ]
            if sub:
                summary[cond][str(length)] = aggregate(sub)
        allc = [r for r in rows if r["condition"] == cond]
        summary[cond]["all"] = aggregate(allc)

    out = args.out or f"results/raw/stage1_baseline_draw{args.draw}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if args.timing_only:
        # Re-measuring latency must not silently discard the quality sweep that took
        # half an hour to produce.
        if not os.path.exists(out):
            raise SystemExit(f"--timing-only needs an existing {out} to update")
        with open(out, encoding="utf-8") as fh:
            prev = json.load(fh)
        prev["timings"] = timings
        prev["timing_restamped"] = stamp(
            trials=args.trials, warmup=args.warmup, note="timings re-measured"
        )
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(prev, fh, indent=1)
        print(f"\nupdated timings in {out} (quality rows preserved)")
        return 0

    payload = {
        "stage": 1,
        "draw": args.draw,
        "model_id": args.model,
        "geometry": tm.geometry.to_json(),
        "n_documents": len(docs),
        "n_rows": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "summary": summary,
        "timings": timings,
        "rows": rows,
        **stamp(),
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print("\n=== Stage 1 summary (clean_hit) ===")
    for cond in ("NATIVE", "NOCTX"):
        a = summary[cond]["all"]
        print(f"{cond:7s} overall clean={a['overall']['clean_hit']:.3f} n={a['overall']['n']}")
        for cls, v in a["by_class"].items():
            print(f"        {cls:12s} clean={v['clean_hit']:.3f} margin={v.get('rank_margin', 0):+.3f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
