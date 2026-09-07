"""Pre-RoPE storage and position re-application.

This is the load-bearing mechanism of the whole project, so it lives on its own and is
tested on its own.

The rule, from C²KV (`docs/RELATED_WORK.md` §1.1): a page's keys are captured **before**
rotary embedding is applied, and rotation is applied at *composition* time using the
position the page actually lands at. Values are untouched by RoPE and are stored as-is.

Why it matters: Stage 2a demonstrated that a post-RoPE key carries its own position in
its rotation — reordering post-RoPE cache tensors is a provable no-op, because attention
is permutation-invariant over keys and the position is *inside* each key. So a page whose
keys were already rotated is permanently dated to its extraction position and cannot be
honestly re-placed. Storing pre-RoPE is what makes independent compilation composable.
"""

from __future__ import annotations

import torch

from transformers.models.qwen3.modeling_qwen3 import rotate_half


def cos_sin_for(model, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for the given absolute positions, from the model's own rotary module.

    Deliberately delegates to `model.model.rotary_emb` rather than recomputing the
    frequencies here: Qwen3 keeps `rope_theta` inside `rope_parameters` and applies an
    `attention_scaling` factor, and a hand-rolled copy would silently drift if either
    changed.
    """
    pos = positions.reshape(1, -1).to(model.device)
    probe = torch.empty(1, dtype=dtype, device=model.device)
    return model.model.rotary_emb(probe, pos)


def apply_rope_to_keys(k_pre: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pre-RoPE keys `[B, H_kv, S, D]` to the positions encoded in cos/sin.

    Mirrors `apply_rotary_pos_emb` exactly, keys only (queries are rotated by the model
    itself during the forward pass, and values are never rotated).
    """
    cos = cos.unsqueeze(1).to(k_pre.dtype)
    sin = sin.unsqueeze(1).to(k_pre.dtype)
    return (k_pre * cos) + (rotate_half(k_pre) * sin)


def position_keys(model, k_pre: torch.Tensor, start: int) -> torch.Tensor:
    """Place a page's keys at absolute positions `[start, start + S)`."""
    positions = torch.arange(start, start + k_pre.shape[2], device=model.device)
    cos, sin = cos_sin_for(model, positions, k_pre.dtype)
    return apply_rope_to_keys(k_pre, cos, sin)


class PreRopeCapture:
    """Capture a frozen Qwen3's own pre-RoPE keys and values, per layer.

    Hooks `self_attn.k_norm` (whose output is exactly the key tensor immediately before
    `apply_rotary_pos_emb`) and `self_attn.v_proj`. Used two ways:

    * as the **identity compiler** — store a chunk's real pre-RoPE KV, re-apply position
      at composition, and check the result reproduces a native prefill. That is the
      ratio-1.0 identity test, and it validates the positional plumbing independently of
      any learned weights.
    * to supply chunk keys/values to the learned extractor's attention.
    """

    def __init__(self, model):
        self.model = model
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self) -> PreRopeCapture:
        for idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            self._handles.append(
                attn.k_norm.register_forward_hook(self._make_hook(idx, self.keys))
            )
            self._handles.append(
                attn.v_proj.register_forward_hook(self._make_hook(idx, self.values, view=True))
            )
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, idx: int, store: dict, view: bool = False):
        head_dim = self.model.config.head_dim

        def hook(_module, _inp, out):
            # k_norm output is already [B, S, H_kv, D]; v_proj output is [B, S, H_kv*D].
            t = out.view(*out.shape[:2], -1, head_dim) if view else out
            store[idx] = t.transpose(1, 2).contiguous()

        return hook

    def stacked(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (pre-RoPE keys, values), each `[B, H_kv, S, D]`."""
        return [(self.keys[i], self.values[i]) for i in sorted(self.keys)]


def compose_pages(
    model,
    pages: list[list[tuple[torch.Tensor, torch.Tensor]]],
    start: int = 0,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
    """Concatenate independently compiled pages, rotating each to where it lands.

    `pages[p][layer] = (k_pre, v)`. Pages are laid out back to back starting at absolute
    position `start`; each page's keys are rotated to the offset it actually occupies,
    which is the entire point of storing them unrotated. Returns per-layer composed
    (k, v) plus the next free position.
    """
    if not pages:
        raise ValueError("compose_pages needs at least one page")
    n_layers = len(pages[0])
    composed_k: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    composed_v: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    pos = start
    for page in pages:
        if len(page) != n_layers:
            raise ValueError("pages disagree on layer count")
        span = page[0][0].shape[2]
        positions = torch.arange(pos, pos + span, device=model.device)
        cos, sin = cos_sin_for(model, positions, page[0][0].dtype)
        for layer, (k_pre, v) in enumerate(page):
            composed_k[layer].append(apply_rope_to_keys(k_pre, cos, sin))
            composed_v[layer].append(v)
        pos += span

    out = [
        (torch.cat(composed_k[i], dim=2), torch.cat(composed_v[i], dim=2))
        for i in range(n_layers)
    ]
    return out, pos


def cache_from_layers(model, layers: list[tuple[torch.Tensor, torch.Tensor]]):
    """Build a `DynamicCache` from per-layer (k, v) tensors."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    for idx, (k, v) in enumerate(layers):
        cache.update(k.contiguous(), v.contiguous(), idx)
    return cache
