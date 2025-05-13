# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import logging
import os
import queue
import subprocess as sp
import sys
import tempfile
import time
from multiprocessing.connection import Client, Connection, Listener
from typing import Optional

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from fastgen.generate import Fastgen, GenArgs, Packet
from fastgen.utils.loading import HfLoader
from fastgen.utils.tokenizer import Message


def worker_main(
    args,
    rdv_dir: str,
    rank: int,
    c: Connection,
) -> None:
    logging.basicConfig(
        format=f"%(asctime)s %(levelname)-1.1s - {rank} - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )

    world_size = args.tensor_parallel
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if world_size == 1:
        mesh: Optional[DeviceMesh] = None
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=args.tensor_parallel,
            init_method=f"file://{rdv_dir}/rdv",
        )
        mesh = init_device_mesh("cuda", (world_size,))

    torch.manual_seed(777)
    torch.set_default_device(device)

    loader = HfLoader(args.model)
    gen_args = GenArgs(
        use_sampling=bool(args.temperature),
        temperature=args.temperature,
        top_p=args.top_p,
        max_batch=1,
        max_gen=args.max_tokens,
        gpu_gb=args.gpu_gb,
    )
    fg = Fastgen.build(
        loader=loader,
        gen_args=gen_args,
        tp_mesh=mesh,
        device=device,
    )
    c.send("ready")

    q: queue.Queue = queue.Queue()
    g = fg.generate(q)

    while True:
        msg, payload = c.recv()
        if msg == "quit":
            break
        if msg == "gen":
            t0 = time.monotonic()
            q.put(Packet(thread_id=0, tokens=payload))
            p = next(g)
            tps = len(p.tokens) / (time.monotonic() - t0)
            if rank == 0:
                c.send((p, tps))

    fg.destroy()
    if mesh is not None:
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="checkpoint directory (Hugging Face)",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="number of gpus to use for parallel inference",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable logging",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="sampling temperature (0.0 for greedy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="top-p samping (1.0 to disable)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="maximum number of tokens per answer",
    )
    parser.add_argument(
        "--gpu-gb",
        type=float,
        default=20,
        help="approximate amount of gpu memory to use",
    )
    parser.add_argument("--worker", type=str, help="for internal use only")
    args = parser.parse_args()

    if args.worker:
        rdv_dir, rank, addr = eval(args.worker)
        return worker_main(args, rdv_dir, rank, Client(addr))

    loader = HfLoader(args.model)
    tokenizer = loader.load_tokenizer()

    # spawn worker processes
    worker_args = [sys.executable] + sys.argv[:]
    worker_args.append("--worker")
    workers: list[tuple[sp.Popen, Connection]] = []
    listener = Listener(("127.0.0.1", 0), "AF_INET")
    with tempfile.TemporaryDirectory() as rdv_dir:
        for rank in range(args.tensor_parallel):
            worker_args.append(repr((str(rdv_dir), rank, listener.address)))
            pg = os.getpgid(os.getpid())
            env = dict(os.environ)
            env["ENABLE_INTRA_NODE_COMM"] = "1"
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
            wp = sp.Popen(worker_args, env=env, process_group=pg)
            wc = listener.accept()
            workers.append((wp, wc))
            worker_args.pop()

        # wait for workers to be ready
        for _, c in workers:
            _ = c.recv()

    print("\u001b[32mReady to chat!\u001b[0m")
    print("  Available commands:")
    print("    /r <N> -- reset conversation before round N")
    print("    /m     -- input a multiline message, end your input")
    print("              with a single line containing '.'")
    print("    /q     -- quit")
    print("")
    print("---")

    dialog: list[Message] = []
    while True:
        round = 1 + len(dialog) // 2
        try:
            msg = input(f"{round}> ").strip()
        except EOFError:
            print("^D")
            break
        match msg[:2]:
            case "/q":
                break

            case "/r":
                try:
                    n = int(msg[2:].strip())
                    dialog = dialog[: 2 * (n - 1)]
                except Exception:
                    print("invalid command")
                continue

            case "/m":
                lines: list[str] = []
                while True:
                    line = input(f"{round}>> ")
                    if line.strip() == ".":
                        break
                    lines.append(line)
                msg = "\n".join(lines)

        dialog.append(Message(role="user", content=msg))
        tokens = tokenizer.encode_dialog(dialog)

        for _, c in workers:
            c.send(("gen", tokens))
        pkt, tps = workers[0][1].recv()
        ntoks = len(pkt.tokens)

        if pkt.tokens[-1] in tokenizer.stop_tokens:
            pkt.tokens = pkt.tokens[:-1]
        msg = tokenizer.decode(pkt.tokens)
        dialog.append(Message(role="assistant", content=msg))

        print(msg)
        print(f"--- [{ntoks} tokens, {tps:.3f}toks/s]")

    for _, c in workers:
        c.send(("quit", None))
    for p, _ in workers:
        p.wait()


if __name__ == "__main__":
    main()
