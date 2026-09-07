#!/usr/bin/env python
"""Stage 1a: verify target model geometry before building anything on top of it.

Checks tokenizer, hidden dim, layer count, attention geometry, KV heads, head dim, RoPE
and the cache implementation. Writes results/raw/model_probe.json.

Weights are loaded only with --load-weights, which additionally verifies the live RoPE
module and the concrete Cache class a real forward pass produces - the two things a
config file cannot tell you and that Stage 2 depends on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ccl.provenance import stamp  # noqa: E402
from ccl.target import read_geometry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--load-weights", action="store_true")
    ap.add_argument("--out", default="results/raw/model_probe.json")
    args = ap.parse_args()

    geo = read_geometry(args.model)
    report: dict = {"geometry": geo.to_json(), **stamp(model_id=args.model)}

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    probe_text = "The replacement assembly is QZ-8471-BX; digest 9f3a1c0d, /var/opt/x.log"
    report["tokenizer"] = {
        "class": type(tok).__name__,
        "vocab_size": tok.vocab_size,
        "len_tokenizer": len(tok),
        "model_max_length": tok.model_max_length,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "has_chat_template": tok.chat_template is not None,
        "probe_text": probe_text,
        "probe_n_tokens": len(tok(probe_text, add_special_tokens=False)["input_ids"]),
        "probe_roundtrip_exact": tok.decode(
            tok(probe_text, add_special_tokens=False)["input_ids"]
        )
        == probe_text,
    }
    chat = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    report["tokenizer"]["chat_template_nothink"] = chat

    # KV budget: the quantity this project exists to shrink.
    report["kv_budget"] = {
        "bytes_per_token_bf16": geo.kv_bytes_per_token,
        **{
            f"gib_at_{n}": round(geo.kv_bytes_per_token * n / 2**30, 4)
            for n in (1024, 2048, 4096, 8192, 16384, 32768)
        },
    }

    if args.load_weights:
        import torch

        from ccl.target import TargetModel

        tm = TargetModel(args.model)
        ids = tm.encode(tm.build_prompt("What is two plus two?", None))
        logits, cache = tm.prefill(ids)
        layer = cache.layers[0]
        keys = getattr(layer, "keys", None)
        report["runtime"] = {
            "device": tm.device,
            "dtype": str(tm.dtype),
            "cache_class": type(cache).__name__,
            "cache_layer_class": type(layer).__name__,
            "n_cache_layers": len(cache.layers),
            "cache_seq_length": cache.get_seq_length(),
            "key_shape": list(keys.shape) if keys is not None else None,
            "key_dtype": str(keys.dtype) if keys is not None else None,
            "logits_shape": list(logits.shape),
            "attn_implementation": getattr(tm.model.config, "_attn_implementation", None),
            "rotary_class": type(getattr(tm.model.model, "rotary_emb", None)).__name__,
            "rope_inv_freq_shape": list(tm.model.model.rotary_emb.inv_freq.shape)
            if hasattr(getattr(tm.model.model, "rotary_emb", None), "inv_freq")
            else None,
            "n_params": sum(p.numel() for p in tm.model.parameters()),
            "memory": tm.memory_report(),
        }
        # Sanity: the model must actually answer a trivial question. If this fails,
        # nothing downstream is interpretable.
        out = tm.generate_greedy(ids, max_new_tokens=12)
        report["runtime"]["smoke_answer"] = out["text"]
        report["runtime"]["smoke_ttft_s"] = out["ttft_s"]
        del tm
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
