# Copyright (c) Meta Platforms, Inc. and affiliates.

import time

import fire
import numpy as np
import torch
from torch.nn.init import xavier_normal_
from torch.profiler import ProfilerActivity, profile
from xformers.ops import fmha
from xformers.ops.fmha import flash
from xformers.ops.fmha.attn_bias import (
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)

# The goal here was to find out if fastgen perf will
# be hurt by recording decoding graphs with the max
# possible sequence length but replaying them with
# shorter sequences.
#
# The answer is: no for big batches.
#
# Here are some concrete grid sizes for different
# configs:
#
#    sl    |   bs   |     grid
#  --------+--------+------------
#    1k    |   128  |  1, 2, 128
#    8k    |   128  |  1, 2, 128
#    256k  |   128  |  1, 2, 128
#  --------+--------+------------
#    1k    |    32  |  1, 8, 32
#    8k    |    32  |  1, 7, 32   (wat?)
#    256k  |    32  |  1, 8, 32
#  --------+--------+------------
#    1k    |     2  |  1, 8, 2
#    8k    |     2  |  1, 64, 2
#    256k  |     2  |  1, 114, 2
#
# The grid will be badly sized only for small batch
# sizes with small sequence lengths; that should not
# be a typical load.
#
# Even for small batch sizes, the overhead incurred
# by an overly large block count is not prohibitive.
# The splitkv forward with batch size 2 goes from
# 6us to 20us if we over-provision blocks (grid:
# 1,114,2) for a sequence length of 1k.
#
# Another observation made is that the grid is
# independent of the exact sequence lengths; only
# the max sequence length in the batch matters.
# (This is a welcome limitation as a different
# behavior would make flash a pain to graph.)


# params here are for llama 3.1 405
N_Q_H = 8
N_KV_H = 1
D_HEAD = 128


def main(
    sl: int,  # sequence length
    bs: int = 1,  # batch size
    fake_sl: int = -1,  # lie to flash on the max seqlen
    page: int = 256,
    warmup: int = 3,
    iter: int = 20,
) -> None:
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    k = torch.empty((1, sl * bs, N_KV_H, 1, D_HEAD))
    v = torch.empty((1, sl * bs, N_KV_H, 1, D_HEAD))

    xavier_normal_(k)
    xavier_normal_(v)

    print(f"kvcache: {(2 * k.numel() * 2) / 1024 / 1024}MiB")

    k = k.expand((1, sl * bs, N_KV_H, N_Q_H // N_KV_H, D_HEAD))
    v = v.expand((1, sl * bs, N_KV_H, N_Q_H // N_KV_H, D_HEAD))

    q = torch.empty((1, bs, N_KV_H, N_Q_H // N_KV_H, D_HEAD))

    xavier_normal_(q)

    print(f"{q.shape=} {k.shape=}")

    block_tables = torch.arange(0, sl // page, dtype=torch.int)
    block_tables = block_tables[None, :].expand(bs, sl // page)
    attn_bias = AttnBias.from_seqlens(
        q_seqlen=[1] * bs,
        kv_seqlen=[sl] + [1] * (bs - 1),
        block_tables=block_tables,
        page_size=page,
    )

    if fake_sl > 0:
        assert sl <= fake_sl
        attn_bias.k_seqinfo.max_seqlen = fake_sl

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _attn = fmha.memory_efficient_attention_forward(
            q,
            k,
            v,
            attn_bias,
            op=flash.FwOp,
        )
        torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            max_name_column_width=300,
            sort_by="self_cuda_time_total",
            row_limit=1,
        )
    )
    prof.export_chrome_trace(f"trace_{sl=}_{bs=}.json")

    timings: list[float] = []
    for n in range(iter):
        t0 = time.monotonic()
        torch.cuda.synchronize()
        _attn = fmha.memory_efficient_attention_forward(
            q,
            k,
            v,
            attn_bias,
            op=flash.FwOp,
        )
        torch.cuda.synchronize()
        if n >= warmup:
            print(".", flush=True, end="")
            timings.append((time.monotonic() - t0) * 1e6)
    print()

    avg = np.average(timings)
    std = np.std(timings)
    print(f"{sl=}  {avg:.03f}us±{std:.06f}")


if __name__ == "__main__":
    fire.Fire(main)
