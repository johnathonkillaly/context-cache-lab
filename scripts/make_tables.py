#!/usr/bin/env python
"""Generate summary tables (CSV + Markdown) from raw Stage 1 results.

Also evaluates the Stage 1 gate from docs/EXPERIMENT.md section 6:
NATIVE must clearly beat NOCTX on every fact class, and NOCTX must sit near the floor on
the exact classes. The gate verdict is printed and written into the report - it is not a
judgement call made in prose afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ccl.corpus import EXACT_CLASSES  # noqa: E402

#: Stage 1 gate thresholds, fixed here rather than eyeballed per run.
GATE_MIN_NATIVE_MINUS_NOCTX = 0.25  # per fact class, clean_hit
GATE_MAX_NOCTX_EXACT = 0.05  # exact classes must be near floor


def _rows_to_csv(rows: list[dict], path: str, fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/raw/stage1_baseline_drawA.json")
    ap.add_argument("--outdir", default="results/tables")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as fh:
        data = json.load(fh)

    rows = data["rows"]
    draw = data["draw"]
    lengths = sorted({r["target_tokens"] for r in rows})
    classes = sorted({r["fact_class"] for r in rows})
    conditions = sorted({r["condition"] for r in rows})

    def sel(cond: str, length: int | None = None, cls: str | None = None) -> list[dict]:
        return [
            r for r in rows
            if r["condition"] == cond
            and (length is None or r["target_tokens"] == length)
            and (cls is None or r["fact_class"] == cls)
        ]

    def rate(sub: list[dict], key: str = "clean_hit") -> float | None:
        return sum(bool(r[key]) for r in sub) / len(sub) if sub else None

    def margin(sub: list[dict]) -> float | None:
        v = [r["rank_margin"] for r in sub if r.get("rank_margin") is not None]
        return sum(v) / len(v) if v else None

    os.makedirs(args.outdir, exist_ok=True)
    _rows_to_csv(
        rows,
        os.path.join(args.outdir, f"stage1_rows_draw{draw}.csv"),
        [
            "condition", "draw", "doc_id", "target_tokens", "n_chunks", "fact_id",
            "fact_class", "gold", "distractor", "prediction", "semantic_hit",
            "exact_hit", "distractor_hit", "clean_hit", "answer_logprob",
            "distractor_logprob", "rank_margin", "n_prompt_tokens",
            "n_active_positions", "n_generated",
        ],
    )

    # -- accuracy by condition x length --------------------------------------------
    acc_rows = []
    for cond in conditions:
        for length in lengths:
            sub = sel(cond, length)
            acc_rows.append({
                "condition": cond, "target_tokens": length, "n": len(sub),
                "clean_hit": rate(sub), "semantic_hit": rate(sub, "semantic_hit"),
                "exact_hit": rate(sub, "exact_hit"),
                "distractor_hit": rate(sub, "distractor_hit"),
                "rank_margin": margin(sub),
            })
    _rows_to_csv(
        acc_rows, os.path.join(args.outdir, f"stage1_by_length_draw{draw}.csv"),
        list(acc_rows[0]),
    )

    # -- accuracy by condition x class ---------------------------------------------
    cls_rows = []
    for cond in conditions:
        for cls in classes:
            sub = sel(cond, None, cls)
            cls_rows.append({
                "condition": cond, "fact_class": cls, "n": len(sub),
                "clean_hit": rate(sub), "semantic_hit": rate(sub, "semantic_hit"),
                "exact_hit": rate(sub, "exact_hit"),
                "distractor_hit": rate(sub, "distractor_hit"),
                "rank_margin": margin(sub),
            })
    _rows_to_csv(
        cls_rows, os.path.join(args.outdir, f"stage1_by_class_draw{draw}.csv"),
        list(cls_rows[0]),
    )

    # -- gate ------------------------------------------------------------------------
    gate: dict = {"per_class": {}, "passed": True, "failures": []}
    for cls in classes:
        nat = rate(sel("NATIVE", None, cls)) or 0.0
        noc = rate(sel("NOCTX", None, cls)) or 0.0
        delta = nat - noc
        ok_sep = delta >= GATE_MIN_NATIVE_MINUS_NOCTX
        ok_floor = (noc <= GATE_MAX_NOCTX_EXACT) if cls in EXACT_CLASSES else True
        gate["per_class"][cls] = {
            "native": nat, "noctx": noc, "delta": delta,
            "separation_ok": ok_sep, "floor_ok": ok_floor,
            "native_margin": margin(sel("NATIVE", None, cls)),
            "noctx_margin": margin(sel("NOCTX", None, cls)),
        }
        if not ok_sep:
            gate["passed"] = False
            gate["failures"].append(
                f"{cls}: NATIVE-NOCTX={delta:.3f} < {GATE_MIN_NATIVE_MINUS_NOCTX}"
            )
        if not ok_floor:
            gate["passed"] = False
            gate["failures"].append(
                f"{cls}: NOCTX={noc:.3f} > {GATE_MAX_NOCTX_EXACT} (corpus contamination)"
            )

    # -- markdown report ---------------------------------------------------------------
    md = [
        f"# Stage 1 — native baseline (draw {draw})",
        "",
        f"Model `{data['model_id']}` · {data['n_documents']} documents · "
        f"{data['n_rows']} evaluations · commit `{data['git_commit']}` · "
        f"{data['timestamp_utc']}",
        "",
        "Generated by `scripts/make_tables.py`. Do not hand-edit.",
        "",
        "## Gate verdict",
        "",
        f"**{'PASS' if gate['passed'] else 'FAIL'}** — "
        f"NATIVE must beat NOCTX by ≥{GATE_MIN_NATIVE_MINUS_NOCTX:.2f} clean_hit on every "
        f"fact class, and NOCTX must stay ≤{GATE_MAX_NOCTX_EXACT:.2f} on exact classes.",
        "",
    ]
    if gate["failures"]:
        md += ["Failures:", ""] + [f"- {f}" for f in gate["failures"]] + [""]

    md += [
        "## Accuracy by fact class (all lengths pooled, `clean_hit`)",
        "",
        _md_table(
            ["fact class", "NATIVE", "NOCTX", "Δ", "NATIVE margin", "NOCTX margin", "gate"],
            [
                [
                    cls,
                    f"{g['native']:.3f}",
                    f"{g['noctx']:.3f}",
                    f"{g['delta']:+.3f}",
                    f"{g['native_margin']:+.2f}" if g["native_margin"] is not None else "-",
                    f"{g['noctx_margin']:+.2f}" if g["noctx_margin"] is not None else "-",
                    "ok" if (g["separation_ok"] and g["floor_ok"]) else "FAIL",
                ]
                for cls, g in gate["per_class"].items()
            ],
        ),
        "",
        "`margin` is mean `logprob(gold) − logprob(distractor)` under teacher forcing. It "
        "detects information that is present but not emitted, which generation accuracy "
        "alone cannot.",
        "",
        "## Accuracy by context length (`clean_hit`)",
        "",
        _md_table(
            ["context", *conditions, "Δ"],
            [
                [
                    f"{length}",
                    *[f"{rate(sel(c, length)) or 0:.3f}" for c in conditions],
                    f"{(rate(sel('NATIVE', length)) or 0) - (rate(sel('NOCTX', length)) or 0):+.3f}",
                ]
                for length in lengths
            ],
        ),
        "",
    ]

    if data.get("timings"):
        md += [
            "## Native prefill cost (cold, no prefix reuse)",
            "",
            _md_table(
                ["context", "prompt tokens", "prefill (s)", "tok/s", "TTFT (s)", "KV (MiB)"],
                [
                    [
                        f"{t['target_tokens']}",
                        f"{t['n_prompt_tokens']}",
                        f"{t['prefill_median_s']:.3f}",
                        f"{t['prefill_tokens_per_s']:.0f}",
                        f"{t['ttft_median_s']:.3f}",
                        f"{t['kv_bytes'] / 2**20:.0f}",
                    ]
                    for t in data["timings"]
                ],
            ),
            "",
            "Median of "
            f"{data['timings'][0]['prefill_stats']['n_trials']} trials after "
            f"{data['timings'][0]['prefill_stats']['n_warmup']} warmups, MPS synchronized. "
            "This is the cost every later stage must beat.",
            "",
        ]

    report = os.path.join(args.outdir, f"stage1_report_draw{draw}.md")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(args.outdir, f"stage1_gate_draw{draw}.json"), "w") as fh:
        json.dump(gate, fh, indent=2)

    print("\n".join(md))
    print(f"\nwrote {report}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
