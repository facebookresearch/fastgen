# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_gpu

try:
    import flash_attn_interface

    _use_flash3 = True
except ModuleNotFoundError:
    _use_flash3 = False

from fastgen.cache import ModelCache, RawCache
from fastgen.kernels import apply_rope, paged_memcpy, rmsnorm
from fastgen.model import RopeArgs
from fastgen.model import Transformer as Model
from fastgen.utils import weights


@dataclass
class SeqInfo:
    """
    Sequence length information for an attention call.
    Queries and keys have possibly different size
    information to allow for key (and value) caching.
    """

    q_seqlen: torch.Tensor
    q_seqstart: torch.Tensor
    q_max_seqlen: int
    k_seqlen: torch.Tensor
    k_max_seqlen: int

    @staticmethod
    def from_seqlen(
        q_seqlen: list[int],
        k_seqlen: list[int],
        device: torch.device,
    ) -> "SeqInfo":
        # cumsum = partial(reduce, lambda out, x: out + [out[-1] + x])
        q_seqstart = [0]
        for seqlen in q_seqlen:
            q_seqstart.append(q_seqstart[-1] + seqlen)
        # k_seqstart = [0]
        # for seqlen in k_seqlen:
        #     k_seqstart.append(k_seqstart[-1] + seqlen)
        int_tensor = partial(
            torch.tensor,
            dtype=torch.int,
            device=device,
        )
        return SeqInfo(
            q_seqlen=int_tensor(q_seqlen),
            q_seqstart=int_tensor(q_seqstart),
            q_max_seqlen=max(q_seqlen),
            k_seqlen=int_tensor(k_seqlen),
            k_max_seqlen=max(k_seqlen),
        )


class ModelState:
    """
    Encapsulates the Transformer's input and state (e.g., caches)
    tensors to simplify using the model in cuda graphs.
    """

    seqinfo: SeqInfo
    tokens: torch.Tensor
    cache: ModelCache
    block_tbl: torch.Tensor
    """
    The block table mapping. That is, an integer tensor of
    shape [B, N] where B is the batch size and N is the
    number of blocks in the cache.

    Note that a block table mapping can address *more*
    tokens than are physically available in the caches;
    that is sensible because the same physical page can
    be reused in different batch lanes.

    The block table mapping can be seen as a tensor that
    tells for each logical lane in the batch what is its
    sequence of physical blocks (pages).
    """
    block_len: int
    batch_size: int
    _actual_batch_size: torch.Tensor

    def __init__(
        self,
        batch_size: int,
        tokens: torch.Tensor,
        block_tbl: torch.Tensor,
        block_len: int,
        cache: RawCache,
        device: torch.device,
    ):
        """
        Initialize the model such that ``decode()`` can be
        called at once.
        """
        assert tokens.device == device
        assert cache.kv_caches[0][0].device == device

        seqlen = block_tbl.shape[1] * block_len
        self.seqinfo = SeqInfo.from_seqlen(
            q_seqlen=[1] * batch_size,
            k_seqlen=[seqlen] * batch_size,
            device=device,
        )
        self.block_tbl = block_tbl[:batch_size]
        self.block_len = block_len
        self._actual_batch_size = torch.tensor(
            batch_size,
            dtype=torch.int64,
            device=device,
        )
        self.tokens = tokens[:batch_size]
        self.batch_size = batch_size
        self.cache = cache

    @property
    def seqlen(self) -> torch.Tensor:
        """
        The sequence lengths in the input batch. Taking into
        account the tokens to be processed by the next model
        call.
        """
        return self.seqinfo.k_seqlen

    def set_actual_batch_size(self, n: int) -> None:
        self._actual_batch_size.fill_(n)

    def copy_inputs(
        self,
        block_tbl: torch.Tensor,
        tokens: torch.Tensor,
        seqlen: torch.Tensor,
    ) -> None:
        if block_tbl.data_ptr() != self.block_tbl.data_ptr():
            self.block_tbl.copy_(block_tbl[: self.batch_size])
        if tokens.data_ptr() != self.tokens.data_ptr():
            self.tokens.copy_(tokens[: self.batch_size])
        if seqlen.data_ptr() != self.seqlen.data_ptr():
            self.seqlen.copy_(seqlen[: self.batch_size])


def all_gather(model: Model, x: torch.Tensor) -> torch.Tensor:
    """
    Gather a tensor of shape (*, n) into a tensor of
    shape (*, tp_size * n).
    """
    if model.tp_size == 1:
        return x

    gather = [torch.empty_like(x) for _ in range(model.tp_size)]
    torch.distributed.all_gather(gather, x, group=model.tp_group)
    return torch.cat(gather, dim=-1)


def all_reduce(model: Model, x: torch.Tensor) -> None:
    """
    Reduce (with sum) the input tensor in place across
    the tensor-parallel group.
    """
    if model.tp_size > 1:
        torch.distributed.all_reduce(x, group=model.tp_group)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seqinfo: SeqInfo,
    block_tbl: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """
    Call the flash varlen forward attention kernel.
    """
    cu_seqlens_q = seqinfo.q_seqstart
    max_seqlen_q = seqinfo.q_max_seqlen
    max_seqlen_k = seqinfo.k_max_seqlen
    seqused_k = seqinfo.k_seqlen

    # extract the page dimension from the caches
    num_pages = key.shape[0] // page_size
    key = key.view(num_pages, page_size, *key.shape[1:])
    value = value.view(num_pages, page_size, *value.shape[1:])

    # there is a significant pytorch cpu overhead to calling
    # _flash_attn_varlen_forward, so we call the internal
    # implementation directly
    if _use_flash3:
        out, *_ = flash_attn_interface._flash_attn_forward(
            query,
            key,
            value,
            None,  # k_new
            None,  # v_new
            None,  # qv
            None,  # out
            cu_seqlens_q,
            None,  # cu_seqlens_k
            None,  # cu_seqlens_k_new
            None,  # seqused_q
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            block_tbl,
            None,  # kv_batch_idx  ??
            None,  # lefpad_k
            None,  # rotary_cos
            None,  # rotary_sin
            None,  # seqlens_rotary
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
            query.shape[-1] ** (-0.5),  # softmax_scale
            causal=True,
            window_size=(-1, -1),
            attention_chunk=0,
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
            sm_margin=0,
        )
    else:
        out, *_ = flash_attn_gpu.varlen_fwd(
            query,
            key,
            value,
            None,
            cu_seqlens_q,
            cu_seqlens_q,  # cu_seqlens_k: unused with paged kernel
            seqused_k,
            None,  # leftpad_k
            block_tbl,
            None,  # alibi_slopes
            max_seqlen_q,
            max_seqlen_k,
            0.0,  # dropout_p
            query.shape[-1] ** (-0.5),  # softmax_scale
            False,  # zero_tensors
            True,  # causal
            -1,  # window_size_left
            -1,  # window_size_right
            0.0,  # softcap (deactivated)
            False,  # return_softmax
            None,
        )
    return out


def vocab_parallel_embedding(
    model: Model,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Map the input token IDs to their embedding. The model
    arg must have its embedding weight evenly on the vocab
    dimension across the model parallel ranks.

    Args:
        model (Model):
            the model from which the tok_embeddings weight
            is read.
        tokens (torch.Tensor):
            the token ids to embed; must an be int tensor
            of shape (S,).

    Returns:
        torch.Tensor:
            the embeddings of the input tokens, of shape
            (S, D) with D the model dimension and dtype
            bfloat16.
    """
    local_vocab_size = model.tok_embeddings.weight.shape[0]
    start_index = model.tp_rank * local_vocab_size
    end_index = start_index + local_vocab_size
    mask = (tokens < start_index) | (tokens >= end_index)
    input = tokens - start_index
    input[mask] = 0
    h = F.embedding(input, model.tok_embeddings.weight)
    h[mask, :] = 0.0
    all_reduce(model, h)
    return h


def quantize_fp8_rowwise(
    x: torch.Tensor,
    scale_clip: Optional[float],
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8 = torch.finfo(dtype)
    row_max = torch.max(torch.abs(x), dim=1)[0]
    if scale_clip is not None:
        row_max = torch.clamp(row_max, min=fp8.eps, max=scale_clip)
    else:
        row_max = torch.clamp(row_max, min=fp8.eps)
    x_scale = (fp8.max / row_max.to(torch.float32))[:, None]
    x_scale[x_scale == float("inf")] = 1.0
    x_fp8 = x * x_scale
    x_fp8 = torch.clamp(x_fp8, min=fp8.min, max=fp8.max)
    return x_fp8.to(dtype), 1 / x_scale


@torch.compile
def ffn_fp8_rowwise(
    h: torch.Tensor,
    ffn: torch.nn.Module,
    scale_clip: Optional[float],
) -> torch.Tensor:
    h_fp8, h_scale = quantize_fp8_rowwise(h, scale_clip)
    x1: torch.Tensor = torch._scaled_mm(  # type: ignore
        h_fp8,
        ffn.w1.weight.T,
        scale_a=h_scale,
        scale_b=ffn.w1.fp8_scale.unsqueeze(0),
        out_dtype=h.dtype,
    )
    x3: torch.Tensor = torch._scaled_mm(  # type: ignore
        h_fp8,
        ffn.w3.weight.T,
        scale_a=h_scale,
        scale_b=ffn.w3.fp8_scale.unsqueeze(0),
        out_dtype=h.dtype,
    )
    x = F.silu(x1, inplace=True).mul_(x3)
    x_fp8, x_scale = quantize_fp8_rowwise(x, scale_clip)
    out: torch.Tensor = torch._scaled_mm(  # type: ignore
        x_fp8,
        ffn.w2.weight.T,
        scale_a=x_scale,
        scale_b=ffn.w2.fp8_scale.unsqueeze(0),
        out_dtype=h.dtype,
    )
    return out


@torch.inference_mode()
def _forward(
    model: Model,
    q_seqlen: Optional[list[int]],
    actual_batch_size: Optional[torch.Tensor],
    token_values: torch.Tensor,
    seqinfo: SeqInfo,
    block_tbl: torch.Tensor,
    block_len: int,
    cache: ModelCache,
    logits_idx: Optional[torch.Tensor],
    prefill: bool = True,
) -> torch.Tensor:
    """
    Forward with the weights in ``model`` and efficient kernels.
    """

    head_dim = model.args.dim // model.args.n_heads
    n_layers = model.args.n_layers
    n_local_heads = model.args.n_heads // model.tp_size
    n_local_kv_heads = max(1, model.args.n_kv_heads // model.tp_size)
    eps = model.args.norm_eps

    if model.rope_freqs is None:
        model.rope_freqs = rope_freqs(head_dim, model.args.rope)

    cache.page_in(0)

    # q_batch maps each token to its batch number
    # q_seqpos maps each token to its position
    q_batch: Optional[torch.Tensor]
    q_seqpos: torch.Tensor

    if prefill:
        assert q_seqlen is not None
        q_batch = torch.tensor(
            sum(([i] * n for i, n in enumerate(q_seqlen)), []),
            dtype=torch.int,
            device=token_values.device,
        )
        k_seqlen = seqinfo.k_seqlen.tolist()
        q_seqpos_list: list[int] = []
        for n, t in zip(k_seqlen, q_seqlen):
            q_seqpos_list.extend(range(n - t, n))
        q_seqpos = torch.tensor(
            q_seqpos_list,
            dtype=torch.int,
            device=token_values.device,
        )
    else:
        q_batch = None
        # use an on-device tensor so that the decoding
        # remains graphable (i.e., does not depend on
        # host memory)
        q_seqpos = seqinfo.k_seqlen - 1

    if model.args.vocab_parallel:
        h = vocab_parallel_embedding(model, token_values)
    else:
        h_parallel = F.embedding(
            token_values,
            model.tok_embeddings.weight,
        )
        h = all_gather(model, h_parallel)

    for i, l in enumerate(model.layers):
        if i + 1 < n_layers:
            # request a cache page-in for the *next* layer so that
            # potential memory transfers may happen concurrently
            # with the current layer
            cache.page_in(i + 1)

        h_in_attn = rmsnorm(h, l.attention_norm.weight, eps)

        wqkv = weights.consecutive(
            l.attention.wq.weight,
            l.attention.wk.weight,
            l.attention.wv.weight,
        )
        if wqkv is not None:
            bias = weights.consecutive(
                l.attention.wq.bias,
                l.attention.wk.bias,
                l.attention.wv.bias,
            )
            assert not l.attention.wq.bias or bias
            xqkv = F.linear(h_in_attn, wqkv, bias)
            xq = xqkv[:, : (n_local_heads * head_dim)]
            xkv = xqkv[:, (n_local_heads * head_dim) :]
            xk, xv = xkv.chunk(2, 1)
        else:
            xq = F.linear(h_in_attn, l.attention.wq.weight, l.attention.wq.bias)
            xk = F.linear(h_in_attn, l.attention.wk.weight, l.attention.wk.bias)
            xv = F.linear(h_in_attn, l.attention.wv.weight, l.attention.wv.bias)

        xq_shape = xq.shape
        xq = xq.view(xq.shape[0], n_local_heads, head_dim)
        xk = xk.view(xk.shape[0], n_local_kv_heads, head_dim)
        xv = xv.view(xv.shape[0], n_local_kv_heads, head_dim)

        cache_k, cache_v = cache.cache_kv(i)

        apply_rope(xq, q_seqpos, model.rope_freqs)
        apply_rope(xk, q_seqpos, model.rope_freqs)
        for x, c in (xk, cache_k), (xv, cache_v):
            paged_memcpy(
                src=x.view(-1, n_local_kv_heads * head_dim),
                dst=c.view(-1, n_local_kv_heads * head_dim),
                page_tbl=block_tbl,
                dst_pos=q_seqpos,
                src_batch=q_batch,
                batch_size=actual_batch_size,
                page_size=block_len,
            )

        # request a cache page out for the current layer right
        # after rope is done populating the kv-cache tensors
        cache.page_out(i)

        xq = xq.view(xq.shape[0], n_local_heads, head_dim)

        attn = attention(xq, cache_k, cache_v, seqinfo, block_tbl, block_len)

        h_out = F.linear(attn.view(xq_shape), l.attention.wo.weight)
        all_reduce(model, h_out)
        h.add_(h_out)

        h_in_ffn = rmsnorm(h, l.ffn_norm.weight, eps)

        if model.args.quantization == "fp8_rowwise" and 0 < i < n_layers - 1:
            h_out = ffn_fp8_rowwise(
                h=h_in_ffn,
                ffn=l.feed_forward,
                scale_clip=model.fp8_scale_clip,
            )
        else:
            w13 = weights.consecutive(
                l.feed_forward.w1.weight,
                l.feed_forward.w3.weight,
            )
            if w13 is not None:
                x13 = F.linear(h_in_ffn, w13)
                x1, x3 = x13.chunk(2, 1)
            else:
                x1 = F.linear(h_in_ffn, l.feed_forward.w1.weight)
                x3 = F.linear(h_in_ffn, l.feed_forward.w3.weight)

            # in-place operations to save 33% of activations
            x_mul = F.silu(x1, inplace=True).mul_(x3)
            h_out = F.linear(x_mul, l.feed_forward.w2.weight)

        all_reduce(model, h_out)
        h.add_(h_out)

    h = rmsnorm(h, model.norm.weight, eps)
    if logits_idx is not None:
        # note: cuda synchronization point (absent during
        # decoding since logits_idx is None then)
        assert prefill
        h = h[logits_idx]
    output: torch.Tensor
    if model.args.tie_embeddings:
        output = model.tok_embeddings.weight
        assert model.args.vocab_parallel
    else:
        output = model.output.weight
    logits_parallel = F.linear(h, output)
    logits = all_gather(model, logits_parallel)
    return logits.float()


def prefill(
    model: Model,
    token_values: torch.Tensor,
    seq_info: list[tuple[int, int]],
    block_tbl: torch.Tensor,
    block_len: int,
    cache: ModelCache,
) -> torch.Tensor:
    """
    Call the model for prompt processing.

    Args:
        model (Model):
            the model object from which weights are pulled.
        token_values (torch.Tensor):
            the concatenated sequence of tokens for the prompts
            to process; the sequence prompts do not have to be
            of same length (e.g., token_values could be of the
            form ``|prompt1|longerprompt2|ompt3|``).
        seq_info (list[tuple[int, int]]):
            sequence information for the prompts to process;
            for each prompt seq_info has a pair of integers
            ``(n0, n1)`` where ``n0`` is the size of the prompt
            prefix for which we already have a kv-cache and
            ``n1`` is the number of tokens to actually process
            (e.g., for the example token_values above we could
            have ``[(0, 7), (0, 13), (2, 5)]``, assuming that
            the last batch element already has a kv-cache for
            "pr").
        block_tbl (torch.Tensor): see ``ModelState.blktbl``.
        block_len (int): the size of a cache block in tokens.
        cache (ModelCache): the kv-cache to read and write.

    Returns:
        torch.Tensor:
            the logits for the last token of each prompt.
    """
    q_seqlen = [n1 for _, n1 in seq_info]
    seqinfo = SeqInfo.from_seqlen(
        q_seqlen=q_seqlen,
        k_seqlen=[n0 + n1 for n0, n1 in seq_info],
        device=token_values.device,
    )
    logits_idx = torch.cumsum(seqinfo.q_seqlen, 0) - 1
    return _forward(
        model,
        q_seqlen,
        None,
        token_values,
        seqinfo,
        block_tbl,
        block_len,
        cache,
        logits_idx,
        prefill=True,
    )


def decode(
    model: Model,
    state: ModelState,
) -> torch.Tensor:
    """
    Call the model to decode one token.

    Args:
        model (Model):
            the model object from which weights are pulled.
        state (ModelState):
            the model inputs; use ``ModelState.copy_inputs``
            and ``ModelState.set_actual_batch_size`` to set
            the model inputs.

    Returns:
        torch.Tensor:
            the logits for each decoded token.
    """
    return _forward(
        model,
        None,
        state._actual_batch_size,
        state.tokens,
        state.seqinfo,
        state.block_tbl,
        state.block_len,
        state.cache,
        logits_idx=None,
        prefill=False,
    )


def rope_freqs(head_dim: int, args: RopeArgs) -> torch.Tensor:
    """
    Precompute frequencies tensor used in RoPE computations.
    """
    pows = torch.arange(0, head_dim, 2).float() / head_dim
    freqs = 1.0 / (args.theta**pows)

    if not args.use_scaled_rope:
        return freqs

    low_freq_factor = args.low_freq_factor
    high_freq_factor = args.high_freq_factor
    old_ctx_len = args.old_context_len
    low_freq_wavelen = old_ctx_len / low_freq_factor
    high_freq_wavelen = old_ctx_len / high_freq_factor
    scaling = args.scale_factor

    for idx, freq in enumerate(freqs):
        wavelen = 2 * math.pi / freq
        if wavelen > low_freq_wavelen:
            freqs[idx] = freq / scaling
        if high_freq_wavelen <= wavelen <= low_freq_wavelen:
            x = old_ctx_len / wavelen - low_freq_factor
            x /= high_freq_factor - low_freq_factor
            freqs[idx] = (1 - x) * freq / scaling + x * freq

    return freqs
