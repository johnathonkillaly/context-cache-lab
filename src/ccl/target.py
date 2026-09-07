"""The frozen target model.

Everything here treats the model as read-only. No stage of this project fine-tunes it;
Stage 2+ trains a sidecar around it. Keeping that boundary in one file makes it obvious
if it is ever violated.

transformers 5.x note: the legacy tuple `past_key_values` format is gone. All cache
manipulation goes through `Cache` objects (`DynamicCache`), which is also what Stage 2
will need in order to splice compiled state in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .timing import sync


def crop_to(cache: DynamicCache, target_len: int) -> None:
    """Truncate `cache` to `target_len` positions.

    transformers >=5.16 deprecates positive `crop` arguments in favour of a negative
    "remove this many" form; this wrapper keeps the call sites reading as absolute
    lengths, which is what the cache-splicing code in later stages actually reasons about.
    """
    excess = cache.get_seq_length() - target_len
    if excess > 0:
        cache.crop(-excess)

SYSTEM_PROMPT = (
    "You answer questions about a provided document. "
    "Reply with only the exact answer value, copied verbatim from the document. "
    "Do not explain, do not add units, do not write a sentence."
)

_DOC_OPEN = "<document>\n"
_DOC_CLOSE = "\n</document>\n\n"


@dataclass
class ModelGeometry:
    """Everything Stage 2 needs in order to build a compatible KV carrier."""

    model_id: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rope_scaling: Any
    max_position_embeddings: int
    sliding_window: Any
    vocab_size: int
    tie_word_embeddings: bool
    torch_dtype: str

    @property
    def gqa_ratio(self) -> float:
        return self.num_attention_heads / self.num_key_value_heads

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, all layers, 2 bytes per bf16 element."""
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * 2

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["gqa_ratio"] = self.gqa_ratio
        d["kv_bytes_per_token"] = self.kv_bytes_per_token
        return d


def read_geometry(model_id: str) -> ModelGeometry:
    """Geometry from config alone - no weights loaded."""
    cfg = AutoConfig.from_pretrained(model_id)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    # transformers 5.x moved `rope_theta` into the `rope_scaling` dict. Stage 2 has to
    # re-apply RoPE at composition time, so reading the right theta is load-bearing:
    # silently using 0.0 (or a 4.x default of 10_000) against a model trained at
    # 1_000_000 would corrupt every composed position.
    rope_scaling = getattr(cfg, "rope_scaling", None) or {}
    rope_theta = getattr(cfg, "rope_theta", None)
    if not rope_theta and isinstance(rope_scaling, dict):
        rope_theta = rope_scaling.get("rope_theta")
    if not rope_theta:
        raise ValueError(f"could not determine rope_theta for {model_id!r}")
    return ModelGeometry(
        model_id=model_id,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        head_dim=head_dim,
        rope_theta=float(rope_theta),
        rope_scaling=rope_scaling or None,
        max_position_embeddings=cfg.max_position_embeddings,
        sliding_window=getattr(cfg, "sliding_window", None),
        vocab_size=cfg.vocab_size,
        tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
        torch_dtype=str(getattr(cfg, "torch_dtype", "unknown")),
    )


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"  # pragma: no cover - CUDA is deliberately not assumed


class TargetModel:
    """Frozen target LLM plus the primitives every stage needs."""

    def __init__(self, model_id: str, device: str | None = None, dtype: Any = torch.bfloat16):
        self.model_id = model_id
        self.device = device or pick_device()
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=None
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.geometry = read_geometry(model_id)

    # -- tokenization ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    def _chat(self, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _user_block(self, question: str, context: str | None, hint: str | None) -> str:
        h = f" ({hint})" if hint else ""
        if context is None:
            return (
                f"Question: {question}\n"
                f"Answer with only the exact value{h}. If you do not know, reply UNKNOWN."
            )
        return (
            f"{_DOC_OPEN}{context}{_DOC_CLOSE}"
            f"Question: {question}\n"
            f"Answer with only the exact value{h}, copied verbatim from the document."
        )

    def build_prompt(
        self, question: str, context: str | None, hint: str | None = None
    ) -> str:
        """Chat-formatted prompt. `context=None` is the NOCTX control."""
        return self._chat(self._user_block(question, context, hint))

    def split_prompt4(
        self, question: str, context: str, hint: str | None = None
    ) -> tuple[str, str, str, str] | None:
        """Split into (chat head, document body, closing tags, question suffix).

        Stage 2a needs to know exactly which cache positions hold the *document*, so it
        can compress that span and leave the chat scaffolding intact. Returns None if
        BPE does not tokenize the four pieces as the concatenation of their token
        sequences, so a misaligned span is never compressed silently.
        """
        full = self.build_prompt(question, context, hint)
        i = full.find(_DOC_OPEN)
        j = full.find(_DOC_CLOSE, i)
        if i < 0 or j < 0:
            return None
        # The newline that opens _DOC_CLOSE belongs to the *body* side of the split:
        # Qwen's BPE merges a sentence-final "." with the following "\n" into one token,
        # so cutting before the newline splits a token and the four pieces no longer
        # tokenize as the concatenation. Moving the cut one character right leaves the
        # prompt string byte-identical and makes the partition exact.
        cut = j + 1
        head = full[: i + len(_DOC_OPEN)]
        body = full[i + len(_DOC_OPEN) : cut]
        tail = full[cut : j + len(_DOC_CLOSE)]
        suffix = full[j + len(_DOC_CLOSE) :]
        enc = lambda s: self.tokenizer(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        if enc(head) + enc(body) + enc(tail) + enc(suffix) != enc(full):
            return None
        return head, body, tail, suffix

    def split_prompt(
        self, question: str, context: str, hint: str | None = None
    ) -> tuple[str, str] | None:
        """Split into (shared document prefix, per-question suffix).

        Lets the harness prefill a long document **once** per condition and reuse its KV
        across that document's questions. Exact prefix reuse is mathematically identical
        to a fresh prefill, so this changes speed only, never a reported quality number.
        Timing measurements deliberately do NOT use this path.

        Returns None when BPE does not tokenize prefix+suffix as the concatenation of
        their token sequences, in which case the caller must fall back to full prefill.
        """
        full = self.build_prompt(question, context, hint)
        marker = _DOC_CLOSE
        cut = full.find(marker)
        if cut < 0:
            return None
        prefix, suffix = full[: cut + len(marker)], full[cut + len(marker) :]
        a = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        b = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]
        c = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        if a + b != c:
            return None
        return prefix, suffix

    # -- forward primitives ------------------------------------------------------------

    def new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.model.config)

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        cache: DynamicCache | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Forward pass appending into `cache`. Returns (last-position logits, cache).

        `position_ids` and `cache_position` are separable on purpose. After document
        positions have been dropped from the cache, *where a token is written* (a
        contiguous slot in the shrunken cache) is no longer *what position it occupies*
        (its original index, whose RoPE phase the surviving keys were encoded with).
        Conflating the two silently re-dates the query relative to the context.
        """
        cache = cache if cache is not None else self.new_cache()
        past = cache.get_seq_length()
        n = input_ids.shape[1]
        if cache_position is None:
            cache_position = torch.arange(past, past + n, device=input_ids.device)
        if position_ids is None:
            position_ids = cache_position
        out = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            position_ids=position_ids.reshape(1, -1),
            cache_position=cache_position,
            use_cache=True,
        )
        return out.logits[:, -1, :], cache

    @torch.inference_mode()
    def decode_from(
        self, last_logits: torch.Tensor, cache: DynamicCache, max_new_tokens: int = 48
    ) -> dict:
        """Greedy decode continuing from `last_logits`. Mutates `cache`."""
        eos = set(self._eos_ids())
        next_id = int(torch.argmax(last_logits, dim=-1)[0])
        generated = [next_id]
        while len(generated) < max_new_tokens and next_id not in eos:
            cur = torch.tensor([[next_id]], device=self.device)
            logits, cache = self.prefill(cur, cache)
            next_id = int(torch.argmax(logits, dim=-1)[0])
            generated.append(next_id)
        text = self.tokenizer.decode(
            [t for t in generated if t not in eos], skip_special_tokens=True
        )
        return {"text": text.strip(), "n_generated": len(generated)}

    @torch.inference_mode()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int = 48) -> dict:
        """Cold prefill + greedy decode, with TTFT measured separately from total."""
        sync()
        t0 = time.perf_counter()
        logits, cache = self.prefill(input_ids)
        _ = int(torch.argmax(logits, dim=-1)[0])
        sync()
        ttft = time.perf_counter() - t0
        out = self.decode_from(logits, cache, max_new_tokens)
        sync()
        out.update(
            ttft_s=ttft,
            total_s=time.perf_counter() - t0,
            n_prompt_tokens=int(input_ids.shape[1]),
        )
        return out

    def _eos_ids(self) -> list[int]:
        gen = getattr(self.model, "generation_config", None)
        eos = getattr(gen, "eos_token_id", None) if gen else None
        if eos is None:
            eos = self.tokenizer.eos_token_id
        return list(eos) if isinstance(eos, (list, tuple)) else [eos]

    # -- teacher-forced scoring --------------------------------------------------------

    @torch.inference_mode()
    def answer_logprob(
        self,
        answer: str,
        *,
        prompt_last_logits: torch.Tensor,
        cache: DynamicCache,
    ) -> dict:
        """Mean per-token logprob of `answer` given a cache holding the full prompt.

        `prompt_last_logits` already predicts the answer's first token, so only
        `answer[:-1]` needs a forward pass. The cache is cropped back to its incoming
        length before returning, so the caller can score several candidates against the
        same prompt without re-prefilling.

        Lower variance than generation accuracy and independent of output formatting,
        which makes it the right instrument for detecting *partial* degradation before
        it becomes a wrong answer.
        """
        ans = self.encode(answer)
        k = int(ans.shape[1])
        base_len = cache.get_seq_length()
        if k == 1:
            all_logits = prompt_last_logits.unsqueeze(1)
        else:
            past = base_len
            pos = torch.arange(past, past + k - 1, device=self.device).unsqueeze(0)
            out = self.model(
                input_ids=ans[:, :-1],
                past_key_values=cache,
                position_ids=pos,
                use_cache=True,
            )
            all_logits = torch.cat([prompt_last_logits.unsqueeze(1), out.logits], dim=1)
            crop_to(cache, base_len)
        lp = torch.log_softmax(all_logits.float(), dim=-1)
        picked = lp.gather(-1, ans.unsqueeze(-1)).squeeze(-1)[0]
        return {
            "mean_logprob": float(picked.mean()),
            "sum_logprob": float(picked.sum()),
            "n_answer_tokens": k,
        }

    # -- memory ------------------------------------------------------------------------

    @staticmethod
    def memory_report() -> dict:
        try:
            return {
                "mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
                "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            }
        except Exception:  # pragma: no cover
            return {}
