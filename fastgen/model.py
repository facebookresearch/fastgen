# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn
from torch.distributed import get_world_size
from torch.distributed.device_mesh import DeviceMesh


@dataclass
class RopeArgs:
    theta: float

    use_scaled_rope: bool = False
    old_context_len: int = 8192
    scale_factor: int = 8
    low_freq_factor: int = 1
    high_freq_factor: int = 4


@dataclass
class ModelArgs:
    dim: int
    ffn_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    qkv_bias: bool
    norm_eps: float
    vocab_size: int
    vocab_parallel: bool
    tie_embeddings: bool

    rope: RopeArgs

    # Llama 3.1 405B quantization
    quantization: Literal["none", "fp8_rowwise"] = "none"
    fp8_scale_clip: Optional[float] = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(dim, device="meta"))


class Attention(nn.Module):
    def __init__(self, tp_size: int, args: ModelArgs) -> None:
        super().__init__()

        head_dim = args.dim // args.n_heads
        mul_kv_heads = max(1, tp_size // args.n_kv_heads)

        self.wq = nn.Linear(
            args.dim,
            (args.n_heads * head_dim) // tp_size,
            bias=args.qkv_bias,
            device="meta",
        )
        self.wk = nn.Linear(
            args.dim,
            (mul_kv_heads * args.n_kv_heads * head_dim) // tp_size,
            bias=args.qkv_bias,
            device="meta",
        )
        self.wv = nn.Linear(
            args.dim,
            (mul_kv_heads * args.n_kv_heads * head_dim) // tp_size,
            bias=args.qkv_bias,
            device="meta",
        )
        self.wo = nn.Linear(
            (args.n_heads * head_dim) // tp_size,
            args.dim,
            bias=False,
            device="meta",
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        tp_size: int,
        dim: int,
        ffn_dim: int,
        ffn_dtype: Optional[torch.dtype],
    ):
        super().__init__()

        self.w1 = nn.Linear(
            dim,
            ffn_dim // tp_size,
            bias=False,
            device="meta",
            dtype=ffn_dtype,
        )
        self.w2 = nn.Linear(
            ffn_dim // tp_size,
            dim,
            bias=False,
            device="meta",
            dtype=ffn_dtype,
        )
        self.w3 = nn.Linear(
            dim,
            ffn_dim // tp_size,
            bias=False,
            device="meta",
            dtype=ffn_dtype,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        tp_size: int,
        args: ModelArgs,
        ffn_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        self.attention = Attention(tp_size, args)
        self.feed_forward = FeedForward(
            tp_size=tp_size,
            dim=args.dim,
            ffn_dim=args.ffn_dim,
            ffn_dtype=ffn_dtype,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)


class VocabEmbedding(nn.Module):
    def __init__(self, tp_size: int, args: ModelArgs) -> None:
        super().__init__()

        nrows = args.vocab_size
        ncols = args.dim
        if args.vocab_parallel:
            assert args.vocab_size % tp_size == 0, args.vocab_size
            nrows //= tp_size
        else:
            ncols //= tp_size

        self.weight = nn.Parameter(torch.empty(nrows, ncols, device="meta"))


class Transformer(nn.Module):
    """
    A strawman model class; the actual forward
    implementation is in ``forward.py``.

    All the weights are initialized on the meta
    device.
    """

    def __init__(
        self,
        args: ModelArgs,
        tp_mesh: Optional[DeviceMesh] = None,
    ) -> None:
        super().__init__()

        self.args = args

        self.tp_mesh = tp_mesh
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group()
            self.tp_rank = tp_mesh.get_local_rank()
            self.tp_size = get_world_size(self.tp_group)
        else:
            self.tp_rank = 0
            self.tp_size = 1

        assert args.n_heads % self.tp_size == 0
        assert args.ffn_dim % self.tp_size == 0

        # memoized tensor of RoPE frequencies to use
        # during inference; set in ``forward.py``
        self.rope_freqs: Optional[torch.Tensor] = None

        self.tok_embeddings = VocabEmbedding(self.tp_size, args)

        self.layers = nn.ModuleList()
        for i in range(args.n_layers):
            ffn_dtype: Optional[torch.dtype] = None
            if args.quantization == "fp8_rowwise":
                if 0 < i < args.n_layers - 1:
                    ffn_dtype = torch.float8_e4m3fn
            self.layers.append(TransformerBlock(self.tp_size, args, ffn_dtype))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        if not args.tie_embeddings:
            self.output = nn.Linear(
                args.dim,
                args.vocab_size // self.tp_size,
                bias=False,
                device="meta",
            )
