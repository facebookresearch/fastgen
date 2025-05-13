# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass

import torch.distributed

from fastgen.model import Transformer as Model


@dataclass
class _ModelParams:
    params: int
    per_token_activation: int
    per_token_cache: int


@dataclass
class MemParams:
    cache_tokens: int
    prefill_tokens: int

    def round_to(self, n: int) -> None:
        self.cache_tokens = _round_to(self.cache_tokens, n)
        self.prefill_tokens = _round_to(self.prefill_tokens, n)


def _round_to(x: int, n: int) -> int:
    return (x + n - 1) // n * n


def _model_params(model: Model) -> _ModelParams:
    model_params = 0
    for p in model.parameters():
        model_params += p.numel() * p.element_size()

    if model.tp_mesh is not None:
        t = torch.tensor(model_params, device="cuda")
        torch.distributed.all_reduce(
            t,
            op=torch.distributed.ReduceOp.MAX,
            group=model.tp_mesh.get_group(),
        )
        model_params = int(t.item())

    ffn_w1 = model.layers[0].feed_forward.w1.weight
    ffn_act = 2 * ffn_w1.shape[0]  # assumes bf16
    hidden_act = 2 * ffn_w1.shape[1]

    # Each layer keeps the skip connection live,
    # then in the ffn we have x1, x3, and h_in_ffn
    # live simultaneously.
    activation = 2 * hidden_act + 2 * ffn_act

    # Cut ourselves some slack.
    activation = int(activation * 1.2)

    attn_wk = model.layers[0].attention.wk.weight
    local_heads_dim = attn_wk.shape[0]
    kv_cache = 2 * 2 * local_heads_dim * len(model.layers)

    return _ModelParams(
        params=model_params,
        per_token_activation=activation,
        per_token_cache=kv_cache,
    )


def mem_params(
    model: Model,
    prefill_gb: float,
    gpu_gb: float,
) -> MemParams:
    """
    Compute memory-related generation parameters.
    """
    p = _model_params(model)
    prefill = int(prefill_gb * 1e9)
    avail = max(prefill, int(gpu_gb * 1e9) - p.params)
    prefill_tokens = prefill // p.per_token_activation
    cache_tokens = (avail - prefill) // p.per_token_cache
    return MemParams(
        cache_tokens=cache_tokens,
        prefill_tokens=prefill_tokens,
    )
