#!/usr/bin/env python
"""Stage 2b: train the memory extractor.

Ratios are **sampled per example** rather than training one model per ratio. This is
C²KV's `C2KV-Dyn` variant, and on this hardware it is the difference between one training
run and four. It also means every reported ratio comes from the same weights, so a
quality difference between ratios cannot be an artefact of one run having trained longer.

Training data is `train_A.json` — draw A's value pools, a different seed, verified to
share no chunk with the draw-A evaluation documents and no fact value with held-out
draw B. Draw B is never touched here.
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

RATIOS = (2.0, 4.0, 8.0, 16.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--corpus", default="results/raw/corpus/train_A.json")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--layer-share", type=int, default=1)
    ap.add_argument("--n-sink", type=int, default=8)
    ap.add_argument("--max-chunks", type=int, default=4)
    ap.add_argument("--val-docs", type=int, default=12)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-examples", type=int, default=16)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="results/raw/stage2b_extractor.pt")
    ap.add_argument("--out", default="results/raw/stage2b_train.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    docs = load_corpus(args.corpus)
    rng.shuffle(docs)
    val_docs, train_docs = docs[: args.val_docs], docs[args.val_docs :]
    train_ex = build_examples(train_docs, rng, min_chunks=2, max_chunks=args.max_chunks)
    val_ex = build_examples(val_docs, random.Random(999), 2, args.max_chunks)
    rng.shuffle(val_ex)
    val_ex = val_ex[: args.val_examples]
    print(
        f"train {len(train_docs)} docs / {len(train_ex)} examples · "
        f"val {len(val_docs)} docs / {len(val_ex)} examples"
    )

    print(f"loading {args.model} ...", flush=True)
    tm = TargetModel(args.model)
    extractor = MemoryExtractor(tm, layer_share=args.layer_share, n_sink=args.n_sink)
    n_train = extractor.n_trainable()
    print(f"trainable: {n_train:,} ({100 * n_train / 4_022_468_096:.1f}% of target)")
    assert_target_frozen(tm, extractor)

    opt = torch.optim.AdamW(extractor.parameters(), lr=args.lr, weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * max(0.05, 1.0 - prog)

    def validate(step: int) -> dict:
        per_ratio: dict[str, float] = {}
        for r in RATIOS:
            ms = [eval_rank_margin(tm, extractor, e, r) for e in val_ex[:8]]
            ms = [m for m in ms if m is not None]
            per_ratio[str(int(r))] = sum(ms) / max(1, len(ms))
        return {"step": step, "val_rank_margin": per_ratio}

    history: list[dict] = []
    h0 = validate(0)
    print(f"[init] val rank_margin {h0['val_rank_margin']}", flush=True)
    history.append(h0)

    t0 = time.time()
    running: list[float] = []
    by_ratio_loss: dict[float, list[float]] = {r: [] for r in RATIOS}

    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ex = train_ex[rng.randrange(len(train_ex))]
        ratio = rng.choice(RATIOS)
        loss, _ = train_step(tm, extractor, ex, ratio)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(extractor.parameters(), args.clip)
        opt.step()
        if step == 1:
            assert_target_frozen(tm, extractor)
        lv = float(loss.detach())
        running.append(lv)
        by_ratio_loss[ratio].append(lv)

        if step % 25 == 0:
            win = running[-25:]
            eta = (time.time() - t0) / step * (args.steps - step) / 60
            print(
                f"  step {step:5d}/{args.steps}  loss {sum(win) / len(win):.4f}  "
                f"lr {lr_at(step):.2e}  {(time.time() - t0) / step:.2f}s/step  "
                f"eta {eta:.0f}m",
                flush=True,
            )
        if step % args.val_every == 0 or step == args.steps:
            h = validate(step)
            h["train_loss"] = sum(running[-100:]) / len(running[-100:])
            h["loss_by_ratio"] = {
                str(int(r)): (sum(v[-50:]) / len(v[-50:]) if v else None)
                for r, v in by_ratio_loss.items()
            }
            history.append(h)
            print(
                f"  step {step:5d}  train_loss {h['train_loss']:.4f}  "
                f"val margin {h['val_rank_margin']}",
                flush=True,
            )
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
            torch.save(
                {
                    "state_dict": extractor.state_dict(),
                    "layer_share": args.layer_share,
                    "n_sink": args.n_sink,
                    "step": step,
                    "config": vars(args),
                },
                args.ckpt,
            )

    payload = {
        "stage": "2b-train",
        "config": vars(args),
        "n_trainable": n_train,
        "n_train_examples": len(train_ex),
        "history": history,
        "final_train_loss": sum(running[-100:]) / len(running[-100:]),
        "minutes": (time.time() - t0) / 60,
        **stamp(),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out} and {args.ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
