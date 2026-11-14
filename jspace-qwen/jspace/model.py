"""Thin model wrapper: residual blocks, final norm, unembedding, tokenisation.

Conventions follow ``anthropics/jacobian-lens``: "layer l" is the *output* of
decoder block l (0-indexed); the target layer is the output of the final block
(the residual just before the final norm); ``unembed`` = final norm + LM head.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn


class LensModel:
    def __init__(self, hf_model: nn.Module, tokenizer: Any | None, *, force_bos: bool = True):
        hf_model.eval()
        for p in hf_model.parameters():
            p.requires_grad_(False)
        self.hf = hf_model
        self.tokenizer = tokenizer
        self.text = hf_model.model
        self.layers: nn.ModuleList = self.text.layers
        self.final_norm: nn.Module = self.text.norm
        self.embed: nn.Module = self.text.embed_tokens
        self.lm_head: nn.Module = hf_model.lm_head
        cfg = hf_model.config.get_text_config() if hasattr(hf_model.config, "get_text_config") else hf_model.config
        self.n_layers: int = cfg.num_hidden_layers
        self.d_model: int = cfg.hidden_size
        self.vocab_size: int = self.lm_head.weight.shape[0]
        self.dtype = self.lm_head.weight.dtype
        self.force_bos = force_bos

    # ------------------------------------------------------------------ text
    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
        # Qwen tokenizers define no BOS; the reference prepends BOS when one
        # exists. Qwen's pretraining docs start with no BOS, so we leave the
        # sequence as-is (the first 16 positions are excluded from fitting).
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if self.force_bos and bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        return torch.tensor([ids[:max_length]], dtype=torch.long)

    def chat(self, user: str, *, assistant_prefill: str | None = None, system: str | None = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        kw = {}
        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, **kw
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if assistant_prefill:
            text = text + assistant_prefill
        return text

    # --------------------------------------------------------------- forward
    def forward(self, input_ids: torch.Tensor, **kw) -> Any:
        return self.text(input_ids=input_ids, use_cache=False, **kw)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(residual.to(self.dtype)))

    _W32: torch.Tensor | None = None

    def unembed_f32(self, residual: torch.Tensor) -> torch.Tensor:
        """Final RMSNorm + LM head in fp32 (avoids bf16 rank ties)."""
        if self._W32 is None:
            self._W32 = self.lm_head.weight.float()
            self._g32 = self.final_norm.weight.float()
            self._eps = float(getattr(self.final_norm, "variance_epsilon", getattr(self.final_norm, "eps", 1e-6)))
        x = residual.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self._eps) * self._g32
        return x @ self._W32.T

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.forward(input_ids)
        return self.lm_head(out.last_hidden_state)

    def unembed_matrix(self) -> torch.Tensor:
        """``A = W_U diag(g)``: the linear part of the readout (final-norm gain
        folded into the unembedding), fp32 ``[vocab, d_model]``."""
        g = self.final_norm.weight.float()
        return self.lm_head.weight.float() * g[None, :]


class Recorder:
    """Forward hooks capturing block outputs. With ``root`` set, that block's
    output is marked ``requires_grad`` so the autograd graph starts there."""

    def __init__(self, model: LensModel, at: Iterable[int], *, root: int | None = None):
        self.model = model
        self.at = sorted(set(at) | ({root} if root is not None else set()))
        self.root = root
        self.acts: dict[int, torch.Tensor] = {}
        self._h: list = []

    def _hook(self, idx: int):
        def fn(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            if idx == self.root:
                t.requires_grad_(True)
            self.acts[idx] = t

        return fn

    def __enter__(self):
        for i in self.at:
            self._h.append(self.model.layers[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []


class Editor:
    """Forward hooks that *replace* block outputs: ``edits[layer](h) -> h'``."""

    def __init__(self, model: LensModel, edits: dict[int, Any]):
        self.model = model
        self.edits = edits
        self._h: list = []

    def _hook(self, idx: int):
        f = self.edits[idx]

        def fn(module, inputs, output):
            if torch.is_tensor(output):
                return f(output)
            return (f(output[0]), *output[1:])

        return fn

    def __enter__(self):
        for i in sorted(self.edits):
            self._h.append(self.model.layers[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []


def load_model(name: str, *, dtype: torch.dtype = torch.bfloat16, random_init: bool = False,
               config_overrides: dict | None = None, threads: int | None = None) -> LensModel:
    """Load an HF causal LM (or a randomly initialised one from its config,
    for offline testing)."""
    import transformers

    if threads:
        torch.set_num_threads(threads)
    if random_init:
        cfg = transformers.AutoConfig.for_model("qwen3", **(config_overrides or {})) if name == "qwen3-like" \
            else transformers.AutoConfig.from_pretrained(name, **(config_overrides or {}))
        torch.manual_seed(0)
        hf = transformers.AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        tok = None
    else:
        hf = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        tok = transformers.AutoTokenizer.from_pretrained(name)
    return LensModel(hf, tok)


@contextlib.contextmanager
def no_grad_if(flag: bool):
    if flag:
        with torch.no_grad():
            yield
    else:
        yield


def band(lo: int, hi: int) -> list[int]:
    return list(range(lo, hi + 1))


def as_list(x: Sequence[int] | int) -> list[int]:
    return [x] if isinstance(x, int) else list(x)
