"""Stage 2b — the learned C²KV-style memory extractor.

Axis A only: `N` ordinary context positions become `M = N/r` learned memory positions.
Precision (Axis B) is Stage 2c and is deliberately not touched here — mixing them would
make any quality change unattributable. See `docs/RELATED_WORK.md` §3.

Design and the eight training-objective questions: `docs/STAGE2B_DESIGN.md`.

What is trainable: a single shared memory-token embedding plus per-layer Q/K/V projection
heads. What is frozen: everything else, including the norms, `o_proj`, the MLPs and the LM
head — the memory tokens are pushed through the *target's own* layers and only their QKV
projections are replaced. That is what keeps the added parameter count near C²KV's
reported ~10%.
"""

from __future__ import annotations

import torch
from torch import nn

from .rope import apply_rope_to_keys, cos_sin_for


def block_spans(n_tokens: int, ratio: float) -> list[tuple[int, int]]:
    """Contiguous chunk-token blocks, one memory token each. `M = ceil(N / ratio)`."""
    if n_tokens <= 0:
        raise ValueError("empty chunk")
    r = max(1, int(round(ratio)))
    m = max(1, -(-n_tokens // r))
    edges = [round(i * n_tokens / m) for i in range(m + 1)]
    return [(edges[i], max(edges[i] + 1, edges[i + 1])) for i in range(m)]


def extraction_mask(
    spans: list[tuple[int, int]], n_tokens: int, n_sink: int, device
) -> torch.Tensor:
    """Boolean `[M, N + M]` mask, True where a memory token may attend.

    Implements C²KV's structured attention flow for the memory side:

    * **block-local extraction** — memory token `j` sees only its own block, plus a
      **sink block** of the first `n_sink` chunk tokens. Stage 2a is direct evidence the
      sink matters: `KV_SINK` beat `KV_STRIDE` at every ratio where either was nonzero.
    * **causal accumulation** — memory token `j` sees memory tokens `<= j`.

    The third constraint, *original-token invariance*, needs no mask entry here: chunk
    tokens are run through an ordinary forward pass that never sees a memory token, so
    the frozen target's own representations are bit-identical to normal.
    """
    m = len(spans)
    mask = torch.zeros(m, n_tokens + m, dtype=torch.bool, device=device)
    if n_sink > 0:
        mask[:, : min(n_sink, n_tokens)] = True
    for j, (lo, hi) in enumerate(spans):
        mask[j, lo:hi] = True
        mask[j, n_tokens : n_tokens + j + 1] = True
    return mask


class MemoryExtractor(nn.Module):
    """Learned sidecar producing pre-RoPE memory KV for one chunk."""

    def __init__(self, target, layer_share: int = 1, n_sink: int = 8, param_dtype=torch.float32):
        super().__init__()
        # Bypass nn.Module.__setattr__ so the 8 GB frozen target can never be registered
        # as a submodule, moved by .to(), or picked up by .parameters().
        object.__setattr__(self, "target", target)
        cfg = target.model.config
        self.n_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.head_dim = cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.n_sink = n_sink
        self.layer_share = max(1, layer_share)
        self.n_groups = -(-self.n_layers // self.layer_share)
        self.param_dtype = param_dtype

        dev = target.device
        # Shared, content-free memory-token embedding (C²KV: all compression tokens share
        # one learnable vector). Initialized at the mean token embedding so the memory
        # stream starts in-distribution rather than at an arbitrary point.
        emb = target.model.get_input_embeddings().weight
        self.memory_token = nn.Parameter(
            emb.mean(dim=0, keepdim=True).detach().to(param_dtype)
        )

        # Per-layer-group QKV heads, initialized FROM the target's own projections. A
        # memory token therefore starts out behaving like an ordinary token and training
        # only has to learn the deviation, which matters a great deal on a small corpus.
        self.q_proj = nn.ModuleList()
        self.k_proj = nn.ModuleList()
        self.v_proj = nn.ModuleList()
        for g in range(self.n_groups):
            src = target.model.model.layers[min(g * self.layer_share, self.n_layers - 1)].self_attn
            for dst_list, src_proj, out_dim in (
                (self.q_proj, src.q_proj, self.n_heads * self.head_dim),
                (self.k_proj, src.k_proj, self.n_kv * self.head_dim),
                (self.v_proj, src.v_proj, self.n_kv * self.head_dim),
            ):
                lin = nn.Linear(self.hidden, out_dim, bias=False, dtype=param_dtype, device=dev)
                with torch.no_grad():
                    lin.weight.copy_(src_proj.weight.to(param_dtype))
                dst_list.append(lin)
        self.to(dev)

    # -- bookkeeping -------------------------------------------------------------------

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _group(self, layer: int) -> int:
        return layer // self.layer_share

    # -- extraction --------------------------------------------------------------------

    def chunk_kv(self, chunk_ids: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Post-RoPE chunk K/V from an ordinary frozen forward pass.

        `torch.no_grad`, deliberately **not** `inference_mode`: inference-mode tensors
        cannot participate in an autograd graph later, and these feed the extractor's
        attention.
        """
        with torch.no_grad():
            n = chunk_ids.shape[1]
            pos = torch.arange(n, device=chunk_ids.device).unsqueeze(0)
            cache = self.target.new_cache()
            self.target.model(
                input_ids=chunk_ids,
                past_key_values=cache,
                position_ids=pos,
                cache_position=pos[0],
                use_cache=True,
            )
        return [(lay.keys.detach(), lay.values.detach()) for lay in cache.layers]

    def extract(
        self, chunk_ids: torch.Tensor, ratio: float
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compile one chunk into per-layer **pre-RoPE** (K, V) memory state.

        A pure function of the chunk: no question, no other chunk, no document position
        is visible here. That is what makes the result reusable and what makes
        question-specific shortcuts unrepresentable (`docs/STAGE2B_DESIGN.md` §2 Q8).
        """
        model = self.target.model
        n = int(chunk_ids.shape[1])
        spans = block_spans(n, ratio)
        m = len(spans)
        dev = chunk_ids.device

        chunk = self.chunk_kv(chunk_ids)
        mask = extraction_mask(spans, n, self.n_sink, dev)
        # Additive mask in the compute dtype; SDPA on MPS is happier with float masks.
        add_mask = torch.zeros(m, n + m, dtype=self.target.dtype, device=dev)
        add_mask.masked_fill_(~mask, torch.finfo(self.target.dtype).min)
        add_mask = add_mask.view(1, 1, m, n + m)

        # C²KV: each memory token takes the position of the LAST chunk token in its block.
        # Used only to run this extraction pass - the stored keys are captured before
        # rotation, so nothing here is baked into the artifact.
        mem_pos = torch.tensor([hi - 1 for _, hi in spans], device=dev)
        cos_m, sin_m = cos_sin_for(model, mem_pos, self.target.dtype)

        h = self.memory_token.to(self.target.dtype).expand(1, m, self.hidden).contiguous()
        out_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            g = self._group(layer_idx)
            normed = layer.input_layernorm(h)
            nd = normed.to(self.param_dtype)

            q = self.q_proj[g](nd).to(self.target.dtype).view(1, m, self.n_heads, self.head_dim)
            k_pre = self.k_proj[g](nd).to(self.target.dtype).view(1, m, self.n_kv, self.head_dim)
            v = self.v_proj[g](nd).to(self.target.dtype).view(1, m, self.n_kv, self.head_dim)

            # Frozen per-head norms, so the produced K lives in the space the target expects.
            q = attn.q_norm(q).transpose(1, 2)
            k_pre = attn.k_norm(k_pre).transpose(1, 2)
            v = v.transpose(1, 2)

            q_rot = apply_rope_to_keys(q, cos_m, sin_m)
            k_rot = apply_rope_to_keys(k_pre, cos_m, sin_m)

            keys = torch.cat([chunk[layer_idx][0], k_rot], dim=2)
            vals = torch.cat([chunk[layer_idx][1], v], dim=2)
            rep = self.n_heads // self.n_kv
            keys = keys.repeat_interleave(rep, dim=1)
            vals = vals.repeat_interleave(rep, dim=1)

            ctx = torch.nn.functional.scaled_dot_product_attention(
                q_rot, keys, vals, attn_mask=add_mask, scale=attn.scaling
            )
            ctx = ctx.transpose(1, 2).reshape(1, m, self.n_heads * self.head_dim)
            h = h + attn.o_proj(ctx)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

            # Stored PRE-RoPE. Position is re-applied at composition time.
            out_layers.append((k_pre, v))

        return out_layers

    def compile_chunks(
        self, chunk_id_list: list[torch.Tensor], ratio: float
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Compile several chunks **independently** — no cross-chunk information."""
        return [self.extract(ids, ratio) for ids in chunk_id_list]


class IdentityCompiler:
    """Ratio-1.0 reference: store the target's own pre-RoPE KV, unchanged.

    Not a compressor. It exists so the positional plumbing — pre-RoPE capture, storage,
    re-application at an arbitrary composed offset — can be validated end-to-end against
    a native prefill *without* any learned weights in the way. If this drifts from
    native, nothing measured about a learned compressor means anything.
    """

    def __init__(self, target):
        self.target = target

    def extract(self, chunk_ids: torch.Tensor, ratio: float = 1.0):
        from .rope import PreRopeCapture

        with PreRopeCapture(self.target.model) as cap, torch.no_grad():
            n = chunk_ids.shape[1]
            pos = torch.arange(n, device=chunk_ids.device).unsqueeze(0)
            cache = self.target.new_cache()
            self.target.model(
                input_ids=chunk_ids,
                past_key_values=cache,
                position_ids=pos,
                cache_position=pos[0],
                use_cache=True,
            )
            return [(k.detach(), v.detach()) for k, v in cap.stacked()]
