# Copyright (c) Meta Platforms, Inc. and affiliates.

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import torch
from torch.profiler import profile, ProfilerActivity

def rope_call(xq, xk, xv, cache_k, cache_v, q_batch, q_seqpos, block_tables):
    xq = torch.ops.fbgemm.rope_qkv_varseq_prefill(
        xq,
        xk,
        xv,
        cache_k.unsqueeze(0),
        cache_v.unsqueeze(0),
        varseq_batch=q_batch,
        varseq_seqpos=q_seqpos,
        theta=1000000,
        num_groups=None,  # for quantization (Optional[int])
        block_tables=block_tables,
        page_size=256,
        varseq_cache_seqpos=None,  # (Optional[Tensor])
        cache_logical_dtype_int=0,  # 0:bf16 1:fp8 2:int4
        rope_scaling=False,
        old_context_len=8192,
        scaling_factor=8.0,
        lo_freq_factor=1.0,
        hi_freq_factor=4.0,
        qparam_k=None,  # quantization param k (Optional[Tensor])
        qparam_v=None,  # quantization param v (Optional[Tensor])
    )


def main():
    B = 30
    S = 1024
    Q_H = 32
    KV_H = 8
    H = 128

    xq = torch.rand((B*S, Q_H, H), device="cuda", dtype=torch.bfloat16)
    xk = torch.rand((B*S, KV_H, H), device="cuda", dtype=torch.bfloat16)
    xv = torch.rand((B*S, KV_H, H), device="cuda", dtype=torch.bfloat16)

    bs = 0
    bs += sum(x.numel() * 2 for x in (xq, xk, xv))  # read
    bs += sum(x.numel() * 2 for x in (xq,))  # write

    NBLK = (B*S + 255) // 256
    L = NBLK * 256
    cache_k = torch.zeros((L, KV_H, H), device="cuda", dtype=torch.bfloat16)
    cache_v = torch.zeros((L, KV_H, H), device="cuda", dtype=torch.bfloat16)

    bs += sum(x.numel() * 2 for x in (cache_k, cache_v))  # write

    # not really what we want but eh
    blocks = torch.arange(NBLK, device="cuda", dtype=torch.int32)
    block_tables = torch.stack((blocks,) * B, 0)

    q_batch = torch.tensor(sum([[i] * S for i in range(B)], []), device="cuda", dtype=torch.int32)
    q_seqpos = torch.tensor(list(range(S)) * B, device="cuda", dtype=torch.int32)

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        rope_call(xq, xk, xv, cache_k, cache_v, q_batch, q_seqpos, block_tables)
        torch.cuda.synchronize()
    p.export_chrome_trace("rope.json.gz")

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            rope_call(xq, xk, xv, cache_k, cache_v, q_batch, q_seqpos, block_tables)

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    g.replay()
    ev_end.record()
    ev_end.synchronize()

    t = ev_start.elapsed_time(ev_end)
    print(f"10 iterations done in {t:.3}ms")
    tpt = (bs * 10) / t
    print(f"throughput: {tpt / 1e6:.2f}GB/s")


if __name__ == "__main__":
    main()
