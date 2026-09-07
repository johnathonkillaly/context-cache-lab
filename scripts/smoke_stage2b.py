#!/usr/bin/env python
"""Stage 2b smoke test: can the sidecar learn at all?

Deliberately an **overfit** experiment on a handful of draw-A documents. Overfitting is
the desired outcome here, not a warning sign - it is the positive control for the
training path. If loss will not drop on data the model is explicitly allowed to
memorize, the architecture or the plumbing is broken and no amount of compute fixes it.

Passing this is the precondition for paying for a real run
(`docs/STAGE2B_DESIGN.md` §3):

  1. answer loss falls substantially from its value at initialization;
  2. rank margin on the training documents improves toward the NATIVE reference (+13.8);
  3. the ratio-1.0 positional identity property still holds.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from ccl.compressor import MemoryExtractor  # noqa: E402
from ccl.corpus import load_corpus  # noqa: E402
from ccl.provenance import stamp  # noqa: E402
from ccl.target import TargetModel  # noqa: E402
from ccl.train import (  # noqa: E402
    assert_target_frozen,
    build_examples,
    eval_rank_margin,
    train_step,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--ratio", type=float, default=4.0)
    ap.add_argument("--docs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--layer-share", type=int, default=1)
    ap.add_argument("--n-sink", type=int, default=8)
    ap.add_argument("--max-chunks", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--out", default="results/raw/stage2b_smoke.json")
    args = ap.parse_args()

    rng = random.Random(0)
    torch.manual_seed(0)

    docs = [
        d
        for d in load_corpus("results/raw/corpus/draw_A.json")
        if d.target_tokens == args.length
    ][: args.docs]
    print(f"loading {args.model} ...", flush=True)
    tm = TargetModel(args.model)
    ex_all = build_examples(docs, rng, min_chunks=2, max_chunks=args.max_chunks)
    print(f"{len(docs)} docs -> {len(ex_all)} examples (overfit set)")

    extractor = MemoryExtractor(
        tm, layer_share=args.layer_share, n_sink=args.n_sink
    )
    n_train = extractor.n_trainable()
    print(f"trainable: {n_train:,} ({100 * n_train / 4_022_468_096:.1f}% of target)")
    assert_target_frozen(tm, extractor)

    opt = torch.optim.AdamW(extractor.parameters(), lr=args.lr, weight_decay=0.0)

    history: list[dict] = []
    margins0 = [eval_rank_margin(tm, extractor, e, args.ratio) for e in ex_all[:6]]
    m0 = sum(x for x in margins0 if x is not None) / max(1, len(margins0))
    print(f"[init] mean rank_margin over {len(margins0)} examples: {m0:+.3f}", flush=True)

    t0 = time.time()
    running: list[float] = []
    for step in range(1, args.steps + 1):
        ex = ex_all[rng.randrange(len(ex_all))]
        loss, n_ans = train_step(tm, extractor, ex, args.ratio)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(extractor.parameters(), 1.0)
        opt.step()
        if step == 1:
            assert_target_frozen(tm, extractor)
        running.append(float(loss.detach()))
        if step % 10 == 0:
            win = running[-10:]
            print(
                f"  step {step:4d}  loss {sum(win) / len(win):.4f}  "
                f"|g| {float(gnorm):.2f}  {(time.time() - t0) / step:.2f}s/step",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            ms = [eval_rank_margin(tm, extractor, e, args.ratio) for e in ex_all[:6]]
            mm = sum(x for x in ms if x is not None) / max(1, len(ms))
            history.append(
                {"step": step, "loss": sum(running[-10:]) / len(running[-10:]),
                 "rank_margin": mm}
            )
            print(f"  step {step:4d}  rank_margin {mm:+.3f}", flush=True)

    first = sum(running[:10]) / max(1, len(running[:10]))
    last = sum(running[-10:]) / max(1, len(running[-10:]))
    verdict = {
        "loss_first10": first,
        "loss_last10": last,
        "loss_drop": first - last,
        "rank_margin_init": m0,
        "rank_margin_final": history[-1]["rank_margin"] if history else None,
        "learns": bool(last < first * 0.7),
    }
    print("\n=== smoke verdict ===")
    print(json.dumps(verdict, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "stage": "2b-smoke",
                "config": vars(args),
                "n_trainable": n_train,
                "history": history,
                "verdict": verdict,
                **stamp(),
            },
            fh,
            indent=2,
        )
    print(f"wrote {args.out}")
    return 0 if verdict["learns"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
