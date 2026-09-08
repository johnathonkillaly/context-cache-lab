#!/usr/bin/env python
"""Stage 2b tables and the FROZEN gate verdict (docs/EXPERIMENT.md §7b).

The verdict is computed, not argued: this script exits non-zero on FAILED so the gate
cannot be talked past in prose afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ccl.metrics import aggregate  # noqa: E402

# --- FROZEN 2026-09-07, before the compressor was written. Do not edit. ---
GATE = {
    "failed_margin_over_budget": 0.05,
    "failed_learned_at_4x": 0.30,
    "promising_learned_at_4x": 0.40,
    "promising_delta_noctx": 0.35,
    "promising_semantic_margin": 2.0,
    "promising_joint_rel_tol": 0.10,
    "strong_recovery_at_4x": 0.70,
    "strong_learned_at_8x": 0.35,
}


def _md(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/raw/stage2b_eval_drawA.json")
    ap.add_argument("--outdir", default="results/tables")
    args = ap.parse_args()

    data = json.load(open(args.results, encoding="utf-8"))
    rows, draw = data["rows"], data["draw"]
    ratios = [r for r in data["ratios"]]
    os.makedirs(args.outdir, exist_ok=True)

    with open(os.path.join(args.outdir, f"stage2b_rows_draw{draw}.csv"), "w",
              newline="", encoding="utf-8") as fh:
        fields = ["condition", "ratio", "doc_id", "target_tokens", "fact_class", "gold",
                  "prediction", "clean_hit", "semantic_hit", "exact_hit",
                  "distractor_hit", "rank_margin", "answer_logprob",
                  "n_active_positions", "n_memory_positions", "n_doc_tokens"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def sel(cond, ratio=None, cls=None):
        return [r for r in rows
                if r["condition"] == cond
                and (ratio is None or r["ratio"] == ratio)
                and (cls is None or r["fact_class"] == cls)]

    def q(cond, ratio=None, cls=None, key="clean_hit"):
        sub = sel(cond, ratio, cls)
        return (sum(bool(r[key]) for r in sub) / len(sub)) if sub else None

    def margin(cond, ratio=None, cls=None):
        sub = [r for r in sel(cond, ratio, cls) if r.get("rank_margin") is not None]
        return (sum(r["rank_margin"] for r in sub) / len(sub)) if sub else None

    def active(cond, ratio=None):
        sub = sel(cond, ratio)
        return (sum(r["n_active_positions"] for r in sub) / len(sub)) if sub else None

    native = q("NATIVE", 1.0) or 0.0
    noctx = q("NOCTX", 1.0) or 0.0

    def recovery(r):
        b = q("BUDGET", r)
        l = q("LEARNED", r)
        if b is None or l is None or native - b <= 0:
            return None
        return (l - b) / (native - b)

    # ---- gate ------------------------------------------------------------------------
    l4, b4 = q("LEARNED", 4.0), q("BUDGET", 4.0)
    beats_anywhere = any(
        (q("LEARNED", r) or 0) > (q("BUDGET", r) or 0) + GATE["failed_margin_over_budget"]
        for r in ratios
    )
    failed_reasons = []
    if not beats_anywhere:
        failed_reasons.append(
            f"LEARNED never exceeds BUDGET by >{GATE['failed_margin_over_budget']} at any ratio"
        )
    if (l4 or 0) < GATE["failed_learned_at_4x"]:
        failed_reasons.append(
            f"LEARNED(4x)={l4:.3f} < {GATE['failed_learned_at_4x']}"
        )
    rnd = q("RANDOM", 4.0)
    if rnd is not None and l4 is not None and abs(l4 - rnd) < 0.05:
        failed_reasons.append(f"LEARNED(4x)={l4:.3f} indistinguishable from RANDOM={rnd:.3f}")

    sem_margin4 = margin("LEARNED", 4.0, "semantic")
    joint4 = q("LEARNED_JOINT", 4.0)
    promising = []
    if (l4 or 0) >= GATE["promising_learned_at_4x"]:
        promising.append(True)
    if (l4 or 0) - noctx >= GATE["promising_delta_noctx"]:
        promising.append(True)
    prom_ok = (
        (l4 or 0) >= GATE["promising_learned_at_4x"]
        and (l4 or 0) - noctx >= GATE["promising_delta_noctx"]
        and (sem_margin4 or -99) >= GATE["promising_semantic_margin"]
        and (
            joint4 is None or l4 is None or joint4 <= 0
            or (joint4 - l4) / max(joint4, 1e-9) <= GATE["promising_joint_rel_tol"]
        )
    )
    rec4 = recovery(4.0)
    l8 = q("LEARNED", 8.0)
    strong_ok = (rec4 is not None and rec4 >= GATE["strong_recovery_at_4x"]) or (
        (l8 or 0) >= GATE["strong_learned_at_8x"]
    )

    verdict = (
        "FAILED" if failed_reasons
        else "STRONGLY PROMISING" if (prom_ok and strong_ok)
        else "PROMISING" if prom_ok
        else "INCONCLUSIVE"
    )

    classes = sorted({r["fact_class"] for r in rows})
    md = [
        f"# Stage 2b — learned compressed context state (draw {draw})",
        "",
        f"Model `{data['model_id']}` · extractor step {data['ckpt_step']} · "
        f"{data['n_documents']} documents at {data['length']} tokens · "
        f"{data['n_rows']} evaluations · commit `{data['git_commit']}` · "
        f"{data['timestamp_utc']}",
        "",
        "Generated by `scripts/make_tables_s2b.py`. Do not hand-edit.",
        "",
        "## Gate verdict (frozen criteria, `docs/EXPERIMENT.md` §7b)",
        "",
        f"### **{verdict}**",
        "",
        "The Stage 2b question is *not* 'does compression beat no-context'. It is:",
        "**does an approximate representation of ALL chunks beat an exact representation "
        "of 1/r of the context?** So `BUDGET` is the comparator.",
        "",
        f"`recovery(r) = (LEARNED − BUDGET) / (NATIVE − BUDGET)`; NATIVE = {native:.3f}, "
        f"NOCTX = {noctx:.3f}.",
        "",
    ]
    if failed_reasons:
        md += ["Failure reasons:", ""] + [f"- {r}" for r in failed_reasons] + [""]

    md += [
        "## Primary comparison (`clean_hit`)",
        "",
        _md(
            ["ratio", "LEARNED", "BUDGET", "Δ vs BUDGET", "recovery", "LEARNED_JOINT",
             "active positions"],
            [
                [
                    f"{r:.0f}×",
                    f"{q('LEARNED', r):.3f}" if q("LEARNED", r) is not None else "-",
                    f"{q('BUDGET', r):.3f}" if q("BUDGET", r) is not None else "-",
                    f"{(q('LEARNED', r) or 0) - (q('BUDGET', r) or 0):+.3f}",
                    f"{recovery(r):+.2f}" if recovery(r) is not None else "-",
                    f"{q('LEARNED_JOINT', r):.3f}" if q("LEARNED_JOINT", r) is not None else "-",
                    f"{active('LEARNED', r):.0f}" if active("LEARNED", r) else "-",
                ]
                for r in ratios
            ],
        ),
        "",
        "## References and controls",
        "",
        _md(
            ["condition", "ratio", "clean_hit", "rank margin", "active positions"],
            [
                [c, f"{rt:.0f}×",
                 f"{q(c, rt):.3f}" if q(c, rt) is not None else "-",
                 f"{margin(c, rt):+.2f}" if margin(c, rt) is not None else "-",
                 f"{active(c, rt):.0f}" if active(c, rt) else "-"]
                for c, rt in [("NATIVE", 1.0), ("NOCTX", 1.0), ("RANDOM", 4.0),
                              ("WRONGPAGE", 4.0), ("SHUFFLE_PAGES", 4.0)]
                if sel(c, rt)
            ],
        ),
        "",
        "## Per fact class (`clean_hit`) — what survives compression, and what breaks first",
        "",
        _md(
            ["fact class", "NATIVE", *[f"LEARNED {r:.0f}×" for r in ratios],
             "BUDGET 4×", "NOCTX"],
            [
                [
                    c,
                    f"{q('NATIVE', 1.0, c):.3f}" if q("NATIVE", 1.0, c) is not None else "-",
                    *[f"{q('LEARNED', r, c):.3f}" if q("LEARNED", r, c) is not None else "-"
                      for r in ratios],
                    f"{q('BUDGET', 4.0, c):.3f}" if q("BUDGET", 4.0, c) is not None else "-",
                    f"{q('NOCTX', 1.0, c):.3f}" if q("NOCTX", 1.0, c) is not None else "-",
                ]
                for c in classes
            ],
        ),
        "",
        "### rank margin per fact class (LEARNED 4× vs references)",
        "",
        _md(
            ["fact class", "NATIVE", "LEARNED 4×", "BUDGET 4×", "NOCTX"],
            [
                [c,
                 f"{margin('NATIVE', 1.0, c):+.2f}" if margin("NATIVE", 1.0, c) is not None else "-",
                 f"{margin('LEARNED', 4.0, c):+.2f}" if margin("LEARNED", 4.0, c) is not None else "-",
                 f"{margin('BUDGET', 4.0, c):+.2f}" if margin("BUDGET", 4.0, c) is not None else "-",
                 f"{margin('NOCTX', 1.0, c):+.2f}" if margin("NOCTX", 1.0, c) is not None else "-"]
                for c in classes
            ],
        ),
        "",
    ]

    path = os.path.join(args.outdir, f"stage2b_report_draw{draw}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(args.outdir, f"stage2b_gate_draw{draw}.json"), "w") as fh:
        json.dump(
            {"verdict": verdict, "failed_reasons": failed_reasons, "native": native,
             "noctx": noctx, "learned_4x": l4, "budget_4x": b4,
             "recovery_4x": rec4, "learned_8x": l8,
             "semantic_margin_4x": sem_margin4, "thresholds": GATE},
            fh, indent=2,
        )
    print("\n".join(md))
    print(f"\nwrote {path}")
    return 1 if verdict == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
