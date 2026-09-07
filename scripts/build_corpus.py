#!/usr/bin/env python
"""Stage 1b: build a deterministic synthetic corpus draw.

Only the tokenizer is loaded (no weights), so this is fast. Draw A is for development
and training; draw B is held out. The script asserts A/B value disjointness whenever
both draws exist on disk - see docs/EXPERIMENT.md section 8.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ccl.corpus import all_values, generate_corpus, load_corpus, save_corpus  # noqa: E402
from ccl.provenance import stamp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", choices=["A", "B"], required=True)
    ap.add_argument("--seed", type=int, default=None, help="default: 11 for A, 977 for B")
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="tokenizer used for lengths")
    ap.add_argument("--lengths", default="1024,2048,4096,8192,16384")
    ap.add_argument("--n-docs", type=int, default=8)
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--facts-per-class", type=int, default=1)
    ap.add_argument("--outdir", default="results/raw/corpus")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else (11 if args.draw == "A" else 977)
    lengths = [int(x) for x in args.lengths.split(",")]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    def count_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    docs = generate_corpus(
        draw=args.draw,
        seed=seed,
        lengths=lengths,
        count_tokens=count_tokens,
        n_docs=args.n_docs,
        chunk_tokens=args.chunk_tokens,
        facts_per_class=args.facts_per_class,
    )

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"draw_{args.draw}.json")
    save_corpus(docs, path)

    by_len: dict[int, list[int]] = {}
    for d in docs:
        by_len.setdefault(d.target_tokens, []).append(d.n_tokens)
    summary = {
        "draw": args.draw,
        "seed": seed,
        "tokenizer_model": args.model,
        "n_documents": len(docs),
        "n_questions": sum(len(d.facts) for d in docs),
        "chunk_tokens": args.chunk_tokens,
        "lengths": {
            str(k): {
                "n_docs": len(v),
                "mean_actual_tokens": round(sum(v) / len(v), 1),
                "min": min(v),
                "max": max(v),
                "n_chunks": docs[0].meta["n_chunks"] if docs else None,
            }
            for k, v in sorted(by_len.items())
        },
        **stamp(),
    }
    for k, v in by_len.items():
        docs_k = [d for d in docs if d.target_tokens == k]
        summary["lengths"][str(k)]["n_chunks"] = docs_k[0].meta["n_chunks"]

    other = os.path.join(args.outdir, f"draw_{'B' if args.draw == 'A' else 'A'}.json")
    if os.path.exists(other):
        overlap = all_values(docs) & all_values(load_corpus(other))
        summary["ab_value_overlap"] = sorted(overlap)
        if overlap:
            raise SystemExit(
                f"FATAL: draws A and B share {len(overlap)} fact values: "
                f"{sorted(overlap)[:5]}. Held-out evaluation would be contaminated."
            )
        summary["ab_disjoint_verified"] = True

    with open(os.path.join(args.outdir, f"summary_{args.draw}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
