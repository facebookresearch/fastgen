# Copyright (c) Meta Platforms, Inc. and affiliates.

from itertools import product

import torch

from fastgen.kernels.rmsnorm import rmsnorm
# from xformers.ops.rmsnorm import rms_norm


def main():
    NITER = 50
    SEQLENS = [1, 3, 4, 16, 128, 256, 1000, 10000, 100000]
    D = 4096
    w = torch.rand(D, dtype=torch.bfloat16)

    for N in SEQLENS:
        print(f"timings for {N=}")
        x = torch.rand((N, D), dtype=torch.bfloat16)

        rmsnorm(x, w, 1e-5)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(NITER):
                rmsnorm(x, w, 1e-5)

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        g.replay()
        ev_end.record()
        ev_end.synchronize()

        t = ev_start.elapsed_time(ev_end)
        print(f"  {NITER} iterations done in {t:.3}ms")
        tpt = (2 * x.nelement() * x.element_size() * NITER) / t
        print(f"  throughput: {tpt / 1e6:.2f}GB/s")


if __name__ == "__main__":
    torch.manual_seed(24)
    with torch.device("cuda"):
        main()
