# Copyright (c) Meta Platforms, Inc. and affiliates.

from itertools import product

import torch

from fastgen.kernels.rope import apply_rope


def rope_pt(
    x: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    cos = torch.cos(pos.outer(freqs))[:, None]
    sin = torch.sin(pos.outer(freqs))[:, None]
    res = x[:, :, ::2]
    ims = x[:, :, 1::2]
    out_res = (res * cos - ims * sin).unsqueeze(-1)
    out_ims = (ims * cos + res * sin).unsqueeze(-1)
    out = torch.cat((out_res, out_ims), -1)
    return out.reshape(x.shape).to(x.dtype)


def main():
    NITER = 50
    SEQLENS = [1, 3, 4, 16, 128, 256, 1000, 10000, 100000]
    H = 8
    D = 128
    freqs = torch.rand(D // 2, dtype=torch.float32)

    for N in SEQLENS:
        print(f"timings for {N=}")
        x = torch.arange(D, dtype=torch.bfloat16)
        x = x[None, None].expand(N, H, -1).contiguous()
        pos = torch.arange(N, dtype=torch.int)

        apply_rope(x, pos, freqs)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(NITER):
                apply_rope(x, pos, freqs)

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


def tests():
    SEQLENS = [1, 3, 4, 5, 8]
    HEADS = [1, 8, 12]
    DIMS = [128, 192]

    for N, H, D in product(SEQLENS, HEADS, DIMS):
        x = torch.arange(D, dtype=torch.bfloat16)
        x = x[None, None].expand(N, H, -1).contiguous()
        pos = torch.arange(N, dtype=torch.int)
        freqs = torch.rand(D // 2, dtype=torch.float32) 

        y = rope_pt(x, pos, freqs)
        apply_rope(x, pos, freqs)
        assert torch.allclose(x, y, atol=5e-4), (x - y).abs().max()

    print("all tests ok")


if __name__ == "__main__":
    torch.manual_seed(24)
    with torch.device("cuda"):
        tests()
        main()
