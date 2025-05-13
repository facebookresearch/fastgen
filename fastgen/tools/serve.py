# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import logging
import os
import subprocess as sp
import sys
import tempfile
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener
from queue import Queue
from threading import Lock, Thread
from typing import Optional
from uuid import uuid4

import torch
from flask import Flask, jsonify, request
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from fastgen.generate import Fastgen, GenArgs, Packet
from fastgen.utils.loading import HfLoader
from fastgen.utils.tokenizer import BaseTokenizer


@dataclass
class Request:
    messages: list[dict[str, str]]
    temperature: float = 0.8
    top_p: float = 0.95  # ignored
    max_tokens: int = 0
    min_tokens: int = 0  # ignored
    model: Optional[str] = None  # ignored
    n: int = 1
    stop: Optional[str] = None  # ignored


logger = logging.getLogger()
app = Flask(__name__)
workers_lock = Lock()
workers: list[tuple[sp.Popen, Connection]] = []
handles: dict[str, Queue[list[int]]] = {}
running: bool = True
tokenizer: BaseTokenizer


def receiver():
    "Receiver thread"
    while running:
        rid, tokens = workers[0][1].recv()
        if (hnd := handles.get(rid)) is None:
            logger.warning(f"Got completion for unknown handle {rid}")
            continue
        hnd.put(tokens)


@app.route("/chat/completions", methods=["POST"])
def handle_completions():
    rq = Request(**request.get_json())
    rid = uuid4().hex
    prompt_tokens = tokenizer.encode_dialog(rq.messages)
    hnd = Queue()
    handles[rid] = hnd
    with workers_lock:
        for _, c in workers:
            c.send(("gen", (rid, rq, prompt_tokens)))
    rsp = {
        "id": rid,
        "choices": [],
        "created": 0,
        "model": rq.model,
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": len(prompt_tokens),
            "total_tokens": len(prompt_tokens),
        },
    }
    for ix in range(rq.n):
        tokens = hnd.get(block=True)
        rsp["usage"]["completion_tokens"] += len(tokens)
        rsp["usage"]["total_tokens"] += len(tokens)
        if tokens[-1] in tokenizer.stop_tokens:
            tokens = tokens[:-1]
        rsp["choices"].append(
            {
                "index": ix,
                "message": {
                    "role": "assistant",
                    "content": tokenizer.decode(tokens),
                },
            }
        )
    del handles[rid]
    return jsonify(rsp)


def worker_enqueuer(c: Connection, q: Queue) -> None:
    "Enqueuer thread."
    while True:
        msg, payload = c.recv()
        if msg == "quit":
            q.put(None)
            return

        if msg == "gen":
            rid, req, toks = payload
            if req.max_tokens <= 0:
                max_gen: Optional[int] = None
            else:
                max_gen = req.max_tokens
            for _ in range(req.n):
                q.put(
                    Packet(
                        thread_id=rid,
                        temperature=req.temperature,
                        max_gen=max_gen,
                        tokens=toks,
                    )
                )


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
        max_batch=args.max_batch,
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

    q: Queue = Queue()

    enqueuer_thread = Thread(target=worker_enqueuer, args=(c, q))
    enqueuer_thread.start()

    for pkt in fg.generate(q):
        if rank == 0:
            c.send((pkt.thread_id, pkt.tokens))

    enqueuer_thread.join()
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
        "--host",
        type=str,
        default="127.0.0.1",
        help="hostname to serve the completions api at (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5678,
        help="port to serve the completions api at (default: 5678)",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="number of gpus to use for parallel inference",
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
        "--max-batch",
        type=int,
        default=256,
        help="maximum batch size to use for inference",
    )
    parser.add_argument(
        "--gpu-gb",
        type=float,
        default=20,
        help="approximate amount of gpu memory to use",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable logging",
    )
    parser.add_argument("--worker", type=str, help="for internal use only")
    args = parser.parse_args()

    if args.worker:
        rdv_dir, rank, addr = eval(args.worker)
        return worker_main(args, rdv_dir, rank, Client(addr))

    loader = HfLoader(args.model)
    global tokenizer
    tokenizer = loader.load_tokenizer()

    # spawn worker processes
    worker_args = [sys.executable] + sys.argv[:]
    worker_args.append("--worker")
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

    receiver_thread = Thread(target=receiver)
    receiver_thread.start()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
