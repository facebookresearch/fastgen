# Copyright (c) Meta Platforms, Inc. and affiliates.

from itertools import product

import torch

from fastgen.kernels.paged_memcpy import paged_memcpy


def main():
    NITER = 50
    DIM = 8 * 128  # GQA, 8 heads
    PAGE_SIZE = 256
    BATCHES = [1, 16]
    SEQLENS = [100, 8000, 16000, 128000]
    
    for B, N in product(BATCHES, SEQLENS):
        N = (N + B - 1) // B
        NPAGES = B * ((N + PAGE_SIZE - 1) // PAGE_SIZE)
        print(f"timings for {B=} {N=}")
        src = torch.rand((B*N, DIM), dtype=torch.bfloat16)
        dst = torch.zeros((NPAGES * PAGE_SIZE, DIM), dtype=torch.bfloat16)
        page = torch.arange(NPAGES, dtype=torch.int).view(B, -1)
        batch = torch.arange(B*N, dtype=torch.int) // N
        dst_pos = torch.arange(N, dtype=torch.int)
        dst_pos = dst_pos[None].expand(B, N).contiguous().reshape(-1)

        paged_memcpy(
            src,
            dst,
            page,
            dst_pos,
            batch,
            None,
            PAGE_SIZE,
        )
        assert torch.allclose(
            src.view(B, N, DIM),
            dst.view(B, -1, DIM)[:, :N],
        )

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(NITER):
                paged_memcpy(
                    src,
                    dst,
                    page,
                    dst_pos,
                    batch,
                    None,
                    PAGE_SIZE,
                )

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        g.replay()
        ev_end.record()
        ev_end.synchronize()

        t = ev_start.elapsed_time(ev_end)
        print(f"  {NITER} iterations done in {t:.3}ms")
        tpt = 2 * (2 * src.nelement() * NITER) / t
        print(f"  throughput: {tpt / 1e6:.2f}GB/s")

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(NITER):
                dst[:B * N].copy_(src)

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        g.replay()
        ev_end.record()
        ev_end.synchronize()

        t = ev_start.elapsed_time(ev_end)
        print(f"  [copy_] {NITER} iterations done in {t:.3}ms")
        tpt_copy = 2 * (2 * src.nelement() * NITER) / t
        print(f"  [copy_] throughput: {tpt_copy / 1e6:.2f}GB/s")

        print(f"  relative perf: {tpt/tpt_copy * 100:.1f}%")


if __name__ == "__main__":
    with torch.device("cuda"):
        main()
