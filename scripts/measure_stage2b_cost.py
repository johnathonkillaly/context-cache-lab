#!/usr/bin/env python
"""Stage 2b cost accounting: cold TTFT, warm TTFT, and the amortization curve.

C²KV reports TTFT with offline extraction excluded. This script does not: `T_compile` is
measured and reported alongside everything else, and the amortized curve shows exactly
how many queries against the same document are needed before compilation pays for itself.

  TTFT_cold = T_compile + T_compose + T_prefill(Z, q) + T_first_token
  TTFT_warm =             T_compose + T_prefill(Z, q) + T_first_token
  amortized(m) = (T_compile + m * TTFT_warm) / m
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from ccl.compressor import MemoryExtractor  # noqa: E402
from ccl.corpus import load_corpus  # noqa: E402
from ccl.provenance import stamp  # noqa: E402
from ccl.rope import cache_from_layers, compose_pages  # noqa: E402
from ccl.target import TargetModel  # noqa: E402
from ccl.timing import measure, sync  # noqa: E402

RATIOS = (2.0, 4.0, 8.0, 16.0)
REUSE_COUNTS = (1, 2, 4, 8, 16, 32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--ckpt", default="results/raw/stage2b_extractor.pt")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="results/raw/stage2b_cost.json")
    args = ap.parse_args()

    doc = next(
        d for d in load_corpus("results/raw/corpus/draw_A.json")
        if d.target_tokens == args.length
    )
    fact = doc.facts[0]
    print(f"loading {args.model} ...", flush=True)
    tm = TargetModel(args.model)
    ck = torch.load(args.ckpt, map_location=tm.device, weights_only=False)
    extractor = MemoryExtractor(tm, layer_share=ck.get("layer_share", 1),
                                n_sink=ck.get("n_sink", 8))
    extractor.load_state_dict(ck["state_dict"])
    extractor.eval()

    head, _body, tail, suffix = tm.split_prompt4(fact.question, doc.text, fact.answer_hint)
    head_ids, chunk_tok = tm.encode(head), [tm.encode(c.text) for c in doc.chunks]
    tail_suffix = torch.cat([tm.encode(tail), tm.encode(suffix)], dim=1)
    native_ids = tm.encode(tm.build_prompt(fact.question, doc.text, fact.answer_hint))

    # Global warm-up: MPS compiles kernels per shape, and without this the first
    # measured configuration absorbs that cost (see AGENTS.md).
    for _ in range(2):
        _, c = tm.prefill(native_ids)
        del c
    sync()

    @torch.no_grad()
    def native_ttft() -> float:
        return tm.generate_greedy(native_ids, max_new_tokens=1)["ttft_s"]

    _, nat = measure(native_ttft, trials=args.trials, warmup=args.warmup)
    _, nat_pf = measure(
        lambda: tm.prefill(native_ids), trials=args.trials, warmup=args.warmup
    )
    print(f"NATIVE  prefill {nat_pf['median_s']:.3f}s  ttft {nat['median_s']:.3f}s "
          f"({native_ids.shape[1]} tokens)", flush=True)

    with torch.no_grad():
        _, head_cache = tm.prefill(head_ids)
        head_layers = [(l.keys.clone(), l.values.clone()) for l in head_cache.layers]
        head_n = head_cache.get_seq_length()
        del head_cache

    results = []
    for ratio in RATIOS:
        with torch.no_grad():
            _, comp = measure(
                lambda: extractor.compile_chunks(chunk_tok, ratio),
                trials=max(2, args.trials // 2), warmup=1,
            )
            pages = extractor.compile_chunks(chunk_tok, ratio)

            def build_cache():
                composed, nxt = compose_pages(tm.model, pages, start=head_n)
                merged = [
                    (torch.cat([head_layers[i][0], composed[i][0]], dim=2),
                     torch.cat([head_layers[i][1], composed[i][1]], dim=2))
                    for i in range(len(composed))
                ]
                return cache_from_layers(tm.model, merged), nxt

            _, compose_stats = measure(build_cache, trials=args.trials, warmup=args.warmup)

            def warm_ttft() -> float:
                import time

                sync()
                t0 = time.perf_counter()
                cache, nxt = build_cache()
                n = tail_suffix.shape[1]
                pos = torch.arange(nxt, nxt + n, device=tm.device)
                cpos = torch.arange(
                    cache.get_seq_length(), cache.get_seq_length() + n, device=tm.device
                )
                logits, cache = tm.prefill(
                    tail_suffix, cache, position_ids=pos, cache_position=cpos
                )
                _ = int(torch.argmax(logits, dim=-1)[0])
                sync()
                dt = time.perf_counter() - t0
                del cache
                return dt

            _, warm = measure(warm_ttft, trials=args.trials, warmup=args.warmup)

        n_active = head_n + sum(p[0][0].shape[2] for p in pages) + tail_suffix.shape[1]
        t_compile = comp["median_s"]
        t_warm = warm["median_s"]
        row = {
            "ratio": ratio,
            "n_active_positions": int(n_active),
            "n_native_positions": int(native_ids.shape[1]),
            "position_compression": native_ids.shape[1] / n_active,
            "t_compile_s": t_compile,
            "t_compose_s": compose_stats["median_s"],
            "ttft_warm_s": t_warm,
            "ttft_cold_s": t_compile + t_warm,
            "ttft_native_s": nat["median_s"],
            "warm_speedup": nat["median_s"] / t_warm,
            "cold_speedup": nat["median_s"] / (t_compile + t_warm),
            "amortized": {
                str(m): (t_compile + m * t_warm) / m for m in REUSE_COUNTS
            },
            "breakeven_queries": None,
        }
        # Smallest m where amortized cost beats a native prefill per query.
        for m in range(1, 513):
            if (t_compile + m * t_warm) / m < nat["median_s"]:
                row["breakeven_queries"] = m
                break
        results.append(row)
        print(
            f"r={ratio:>4.0f}x  active {n_active:>5} ({row['position_compression']:.1f}x)  "
            f"compile {t_compile:.2f}s  warm TTFT {t_warm:.3f}s "
            f"({row['warm_speedup']:.1f}x)  cold {row['ttft_cold_s']:.2f}s  "
            f"breakeven m={row['breakeven_queries']}",
            flush=True,
        )
        del pages

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {"stage": "2b-cost", "doc_id": doc.doc_id, "length": args.length,
             "model_id": args.model, "ckpt_step": ck.get("step"),
             "native": {"prefill_s": nat_pf["median_s"], "ttft_s": nat["median_s"],
                        "n_tokens": int(native_ids.shape[1])},
             "reuse_counts": list(REUSE_COUNTS), "results": results, **stamp()},
            fh, indent=2,
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
