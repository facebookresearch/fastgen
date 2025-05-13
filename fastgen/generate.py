# Copyright (c) Meta Platforms, Inc. and affiliates.

import bisect
import os
import queue
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from logging import getLogger
from typing import Any, Generator, Iterable, Literal, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.functional import cross_entropy
from torch.profiler import ProfilerActivity, profile

from fastgen import forward as fwd
from fastgen.cache import Cache, DynCache, DynCacheLane, RawCache
from fastgen.model import Transformer as Model
from fastgen.utils import sampling, tune
from fastgen.utils.iset import ISet
from fastgen.utils.loading import BaseLoader
from fastgen.utils.misc import hashints

logger = getLogger()


@dataclass
class GenArgs:
    use_sampling: bool = True
    temperature: float = 0.7
    top_p: float = 0.9

    logprobs: bool = False

    max_batch: int = 256
    max_seq: int = -1  # negative for auto tune
    max_gen: Optional[int] = None

    host_cache_gb: float = 0  # 0 to disable
    cache_block: int = 256
    num_cuda_graphs: int = 16

    # prefill bounds, in tokens
    max_prefill: int = 8192  # negative for auto tune
    min_prefill: int = 2048
    min_prefill_batch: int = 1

    prefill_gb: float = 0.5  # used by auto tune
    gpu_gb: float = 20  # ditto

    fp8_scale_clip: Optional[float] = 1200

    sync_freq: int = 10
    # frequency, in decoding iterations, at which
    # generation lanes are checked for termination;
    # cuda streams are synchronized at this point


@dataclass
class Packet:
    thread_id: Any
    "An arbitrary identifier to link inputs to outputs."
    tokens: list[int] = field(default_factory=list)
    "The prompt if used as input, or else the generation output."
    temperature: Optional[float] = None
    "An optional per-generation temperature setting."
    logprobs: Optional[list[float]] = None
    """
    Logprobs for the tokens list. In output packets only,
    and only when ``GenArgs.logprobs`` is set.
    """
    max_gen: Optional[int] = None
    "The maximum number of tokens to generate."


@dataclass
class DecodeGraph:
    state: fwd.ModelState
    graph: Optional[torch.cuda.CUDAGraph]
    logits: Optional[torch.Tensor]


@dataclass
class Lane(Packet):
    maxlen: int = 0
    prompt: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    blockset: ISet = field(default_factory=ISet)
    prompt_hash: int = 0

    @staticmethod
    def from_pkt(
        p: Packet,
        max_seq: int,
        max_gen: Optional[int],
    ) -> "Lane":
        max_gen = p.max_gen or max_gen
        if max_gen is None:
            maxlen = max_seq
        else:
            maxlen = len(p.tokens) + max_gen
            maxlen = min(maxlen, max_seq)
        return Lane(
            thread_id=p.thread_id,
            temperature=p.temperature,
            max_gen=max_gen,
            prompt=p.tokens,
            maxlen=maxlen,
            prompt_hash=hashints(p.tokens),
        )

    def all_tokens(self) -> list[int]:
        "The prompt and generated tokens."
        return self.prompt + self.tokens

    def add_block(self, bid: int) -> None:
        self.blockset.add(bid)
        self.blocks.append(bid)

    def free(self, free_set: ISet) -> None:
        free_set |= self.blockset
        self.blockset.clear()
        self.blocks = []

    def set_blocks(self, bset: ISet) -> None:
        assert not self.blockset
        self.blockset = bset
        self.blocks = bset.tolist()

    def __len__(self) -> int:
        "Length in tokens, including the prompt."
        return len(self.prompt) + len(self.tokens)


class Fastgen:
    @staticmethod
    def build(
        loader: BaseLoader,
        gen_args: GenArgs,
        tp_mesh: Optional[DeviceMesh],
        device: torch.device,
        profile_sig: Optional[int] = signal.SIGUSR2,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Fastgen":
        assert device.type == "cuda", f"invalid {device=}"
        model_args = loader.load_model_args()
        if gen_args.fp8_scale_clip is not None:
            model_args.fp8_scale_clip = gen_args.fp8_scale_clip
        tokenizer = loader.load_tokenizer()
        if model_args.vocab_size == -1:
            model_args.vocab_size = tokenizer.vocab_size
        start_time = time.monotonic()
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        model = Model(model_args, tp_mesh).to_empty(device=device)
        torch.set_default_dtype(default_dtype)
        loader.load(model)
        t = time.monotonic() - start_time
        logger.info(f"Built and loaded model in {t:.02f}s")
        stop_tokens = tokenizer.stop_tokens
        fg = Fastgen(gen_args, model, stop_tokens, dtype, device)
        if profile_sig is not None:
            if signal.getsignal(profile_sig) is not signal.SIG_DFL:
                logger.warning(
                    f"Signal handler for {profile_sig} is set;"
                    " Fastgen profile handler was not installed"
                )
            else:
                signal.signal(profile_sig, fg._profile_sig_handler)
        return fg

    def __init__(
        self,
        args: GenArgs,
        model: "Model",
        stop_tokens: list[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.device = device
        self.generating = False
        self.last_used_gb: Optional[float] = None

        self.profile_state: Literal[
            "inactive",
            "requested",
            "active",
        ] = "inactive"
        self.profile_iter = 0
        self.profile_iters = int(os.environ.get("FG_PROFILE_ITERS", "8"))
        self.profile_dir = os.environ.get("FG_PROFILE_DIR", "/tmp")
        self.profile_ctx: Any = None
        self.profile_obj: Any = None

        self.model = model
        if model.tp_size > 1:
            r = int(torch.rand(1)[0] * 0x7FFFFFFF)
            self._check_consistent(r, "random seed")

        logger.info("Initializing generator")
        self._log_cuda_mem()

        prefill_gb = max(args.prefill_gb, 0.1)
        mem_params = tune.mem_params(model, prefill_gb, args.gpu_gb)
        mem_params.round_to(args.cache_block)

        assert args.sync_freq < args.cache_block

        self.gen_args = args
        self.stop_tokens = stop_tokens
        self.gen_logprobs = args.logprobs
        self.block_len = args.cache_block
        self.max_batch = args.max_batch
        self.min_prefill_batch = args.min_prefill_batch
        if self.min_prefill_batch >= self.max_batch:
            self.min_prefill_batch = 1
        self.max_prefill = args.max_prefill
        if self.max_prefill < 0:
            self.max_prefill = mem_params.prefill_tokens
            logger.info(f"Setting max prefill length to {self.max_prefill}")
        self.max_seq = args.max_seq
        if self.max_seq > 0 and self.max_seq % self.block_len != 0:
            self.max_seq = tune._round_to(self.max_seq, self.block_len)
        if self.max_seq < 0:
            self.max_seq = mem_params.cache_tokens
            logger.info(f"Setting max sequence length to {self.max_seq}")
        if self.max_seq < self.block_len:
            raise RuntimeError(
                "Not enough memory available for the generation caches",
            )
        if self.max_seq < self.max_batch * self.block_len:
            self.max_batch = self.max_seq // self.block_len
            logger.info("Adjusting max batch to {self.max_batch}")

        nblocks = self.max_seq // self.block_len
        self.cache = RawCache.build(
            model=model,
            length=nblocks * self.block_len,
            dtype=dtype,
            device=device,
        )
        bytes_per_tok = self.cache.bytes_per_token()
        self.cache_ready: Optional[torch.cuda.Event] = None
        self.host_cache = Cache(
            limit_toks=int(args.host_cache_gb * 1e9 / bytes_per_tok),
            node_len=self.block_len,
        )
        logger.info(f"Cache bytes per token: {bytes_per_tok}")
        logger.info(f"Host cache node count limit: {self.host_cache.limit}")
        self.host_cache.preallocate(gpu_cache=self.cache)

        logger.info("Allocated kv cache and host cache")
        self._log_cuda_mem()

        self.parking = deque[Lane]()

        # decoder device tensors; will be copied
        # in the model state before decode calls
        self.nactive = 0
        self.tokens = torch.randint(
            low=0,
            high=model.args.vocab_size,
            size=(self.max_batch,),
            dtype=torch.int,
            device=device,
        )
        self.seqlen = torch.zeros(
            self.max_batch,
            dtype=torch.int,
            device=device,
        )
        self.maxlen = torch.zeros(
            self.max_batch,
            dtype=torch.int,
            device=device,
        )
        self.temps = torch.zeros(
            self.max_batch,
            dtype=torch.float,
            device=device,
        )
        self.block_tbl = torch.zeros(
            (self.max_batch, nblocks),
            dtype=torch.int,
            device=device,
        )

        self.free_blocks = ISet.interval(0, nblocks)

        # per-lane host data
        self.lane = [Lane("dead") for _ in range(self.max_batch)]

        self.use_graphs = not bool(os.environ.get("FG_NO_CUDA_GRAPHS"))
        if not self.use_graphs:
            logger.warning("Fastgen is not using graphs")

        # decide the graph batch sizes
        num_graphs = max(2, args.num_cuda_graphs)
        batch_sizes: list[int] = sorted(
            set(
                torch.linspace(
                    start=1,
                    end=self.max_batch,
                    steps=num_graphs,
                    dtype=torch.float32,
                    device="cpu",
                )
                .round()
                .int()
                .tolist()
            )
        )
        self.graph_batch_sizes = batch_sizes

        # create the graphs
        graphs: list[DecodeGraph] = []
        pool = None
        for bs in reversed(batch_sizes):
            state = fwd.ModelState(
                bs,
                self.tokens,
                self.block_tbl,
                self.block_len,
                self.cache,
                device,
            )

            if self.use_graphs:
                # let triton compile the kernels
                fwd.decode(self.model, state)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    logits = fwd.decode(self.model, state)
                if pool is None:
                    pool = graph.pool()

            else:
                graph = None
                logits = None

            graphs.append(DecodeGraph(state, graph, logits))

        self.graphs = list(reversed(graphs))

        if self.use_graphs:
            logger.info(f"Created {len(self.graphs)} decoding graphs")
            self._log_cuda_mem()

        logger.info("Done initializing")

    def destroy(self) -> None:
        """
        Free some internal data; it was observed to
        be necessary to avoid hangs during
        ``torch.distributed.destroy_process_group()``
        """
        while self.graphs:
            self.graphs.pop()

    def request_profile(self) -> None:
        """
        Request the generation of a profile trace.
        """
        if self.profile_state == "inactive":
            logger.info("Requesting profiling trace...")
            self.profile_state = "requested"

    def _profile_sig_handler(self, _signum, frame):
        self.request_profile()

    def _profile_step(self) -> None:
        if self.profile_state == "active":
            self.profile_iter += 1
            if self.profile_iter == self.profile_iters:
                self._profile_done()
            return

        assert self.profile_state == "requested"
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        # activities = [ProfilerActivity.CUDA]  # useful for debugging
        self.profile_iter = 0
        self.profile_ctx = profile(activities=activities, with_stack=True)
        self.profile_obj = self.profile_ctx.__enter__()
        self.profile_state = "active"

    def _profile_done(self, dump: bool = True) -> None:
        if self.profile_state != "active":
            return
        self.profile_state = "inactive"
        try:
            self.profile_ctx.__exit__(None, None, None)
            if not dump:
                return
            tstamp = datetime.now().strftime("%H%M%S.%f")
            dumpf = f"{self.profile_dir}/fastgen_{os.getpid()}_{tstamp}.json.gz"
            self.profile_obj.export_chrome_trace(dumpf)
            logger.info(f"Profile chrome trace exported to {dumpf}")
        except Exception as e:
            logger.warning(f"Error while exporting profile trace: {e!r}")

    def generate(
        self,
        q: queue.Queue[Optional[Packet]],
    ) -> Generator[Packet, None, None]:
        """
        Read from the queue and generate a completion or an error
        for each prompt. The specific error type returned depends
        on the implementation of the generator.

        The outputs may come in a different order from the inputs
        so that latency can be kept minimal. We use a queue to pass
        requests in order to keep generation running even when the
        input lags. To terminate the generation, send None in the
        queue; in-flights generations will be completed before
        termination.
        """
        assert not self.generating, "multiple generators cannot run concurrently"
        self.generating = True
        yield from self._generate(q)
        self._profile_done()
        self.generating = False

    @torch.inference_mode()
    def _generate(
        self,
        q: queue.Queue[Optional[Packet]],
    ) -> Generator[Packet, None, None]:
        eoq = False
        while not eoq or self.parking or self.nactive > 0:
            if self.profile_state != "inactive":
                self._profile_step()

            done = []
            tokens_list = []
            logprobs_list = []

            self._prepare_lanes()

            if self.nactive > 0:
                tokens, logprobs = self._decode()
                toolong = self.seqlen >= self.maxlen
                toolong = toolong[: self.nactive]
                done += self._ended_lanes(tokens, toolong)
                tokens_list += tokens.tolist()
                logprobs_list += logprobs

            self._sync_host_cache()
            self.host_cache.maintain()

            eoq, added = self._add_lanes(eoq, q)
            if added:
                tokens, logprobs, toolong = self._prefill(added)
                done += self._ended_lanes(tokens, toolong)
                tokens_list += tokens.tolist()
                logprobs_list += logprobs

            # iterate in reverse so we can kill multiple
            # lanes; see _kill_lane()
            for i in reversed(range(self.nactive)):
                self.lane[i].tokens += tokens_list[i]
                if self.gen_logprobs:
                    lp = self.lane[i].logprobs or []
                    lp += logprobs_list[i]
                    self.lane[i].logprobs = lp
                if done[i]:
                    yield self._trim_eos(self.lane[i])
                    # _kill_lane() will shuffle lanes and may
                    # clobber pending copies to host memory;
                    # wait for coherence before continuing
                    self._sync_host_cache()
                    self._kill_lane(i)

    def _prepare_lanes(self) -> None:
        """
        Prepare the lanes to make sure they have enough
        blocks allocated for a decoding call. During
        preparation, some active lanes may be parked
        to free cache blocks.
        """
        sync_freq = self.gen_args.sync_freq
        block_len = self.block_len

        idx = 0
        while idx < self.nactive:
            lane = self.lane[idx]

            avail = len(lane.blocks) * block_len
            if len(lane) + sync_freq <= avail:
                idx += 1
                continue
            # else, we need one more block

            if not self.free_blocks:
                last = self.nactive - 1
                if last > 0:
                    self._kill_lane(last, park=True)
                    if idx == last:
                        break

            if self.free_blocks:
                bid = self.free_blocks.popleft()
                self.block_tbl[idx, len(lane.blocks)] = bid
                lane.add_block(bid)
            else:
                # it is possible that there are no free blocks;
                # but in this case nactive must be 1 and we're
                # going to hit maxlen (~ max_seq) pretty soon
                assert self.nactive == 1
                idx = self.nactive

            avail = len(lane.blocks) * block_len
            assert min(self.max_seq, len(lane) + sync_freq) <= avail

        logger.info(f"{self.nactive} active lanes")

    def _add_lanes(
        self,
        eoq: bool,
        q: queue.Queue[Optional[Packet]],
    ) -> tuple[bool, int]:
        """
        Add new lanes to the ``self.lane`` list.

        Returns:
            tuple[bool, int]:
                A boolean indicating if the queue has run
                out of packets and the number of lanes
                added.
        """
        idx = self.nactive
        buffer = 128  # in number of tokens
        addend = buffer + self.block_len - 1
        avail = len(self.free_blocks)

        lanes: list[Lane] = []
        while idx < self.max_batch:
            lane: Optional[Lane] = None

            if parked := bool(self.parking):
                lane = self.parking.popleft()
            elif not eoq:
                try:
                    pkt = q.get(block=(idx == 0))
                except queue.Empty:
                    break
                if pkt is None:
                    eoq = True
                else:
                    lane = Lane.from_pkt(
                        pkt,
                        self.max_seq,
                        self.gen_args.max_gen,
                    )

            if lane is None:
                break

            ask = (len(lane) + addend) // self.block_len
            if ask > avail:
                if parked:
                    self.parking.appendleft(lane)
                else:
                    self.parking.append(lane)
                break

            idx += 1
            avail -= ask
            lanes.append(lane)

        if (mesh := self.model.tp_mesh) is not None:
            # force all tp ranks to read the same number
            # of packets from the input queue
            min_idx_tensor = self.tensor([idx])
            torch.distributed.all_reduce(
                min_idx_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=mesh.get_group(),
            )
            idx = int(min_idx_tensor[0])
            while idx - self.nactive < len(lanes):
                self.parking.appendleft(lanes[-1])
                lanes = lanes[:-1]

        cksum = int(eoq)
        tokens = 0
        for n, lane in enumerate(lanes):
            ask = (len(lane) + addend) // self.block_len
            assert len(self.free_blocks) >= ask
            blocks, self.free_blocks = self.free_blocks.take(ask)
            lane.set_blocks(blocks)
            self.lane[self.nactive + n] = lane
            # TODO: we may want to hash prompt+tokens
            cksum = hashints([cksum, lane.prompt_hash])
            tokens += len(lane)

        if tokens < self.gen_args.min_prefill:
            if len(lanes) < self.min_prefill_batch:
                if not eoq and self.nactive > 0:
                    # wait some more before prefilling
                    while idx > self.nactive:
                        idx -= 1
                        lane = self.lane[idx]
                        lane.free(self.free_blocks)
                        self.parking.appendleft(lane)

        if cksum:
            self._check_consistent(cksum, "input queue")

        return eoq, idx - self.nactive

    def _check_consistent(self, val: int, what: str) -> None:
        "The ``val`` argument must be a representable uint32."
        if self.model.tp_mesh is None:
            return
        min_val = self.tensor([val])
        torch.distributed.all_reduce(
            min_val,
            op=torch.distributed.ReduceOp.MIN,
            group=self.model.tp_mesh.get_group(),
        )
        if int(min_val[0]) != val:
            raise RuntimeError(f"Inconsistent {what} across mp ranks")

    def _sync_host_cache(self) -> None:
        """
        Ensure that the host cache is coherent before
        returning.
        """
        if self.cache_ready is not None:
            self.cache_ready.wait()
            self.cache_ready = None

    def _ended_lanes(
        self,
        tokens: torch.Tensor,
        toolong: torch.Tensor,
    ) -> list[int]:
        has_eos = torch.zeros_like(toolong)
        for st in self.stop_tokens:
            has_eos = torch.logical_or(has_eos, (tokens == st).sum(1))
        return (has_eos | toolong).tolist()

    def _trim_eos(self, pkt: Packet) -> Packet:
        """
        Trim an answer in place so that it is cut after
        the eos token. The input packet is returned.
        """
        trim = len(pkt.tokens)
        n = max(0, len(pkt.tokens) - self.gen_args.sync_freq - 1)
        for st in self.stop_tokens:
            if st in pkt.tokens[n:]:
                trim = min(trim, pkt.tokens.index(st, n) + 1)
        if pkt.max_gen is not None:
            trim = min(trim, pkt.max_gen)
        pkt.tokens = pkt.tokens[:trim]
        if pkt.logprobs is not None:
            pkt.logprobs = pkt.logprobs[:trim]
        return pkt

    def _decode(self) -> tuple[torch.Tensor, list[list[float]]]:
        """
        Decode from active lanes. The number of decoding
        iterations performed is limited by the sync
        frequency. Decoding iterations will not sync the
        cuda streams.

        Returns:
            tuple[torch.Tensor, list[list[float]]]:
                The decoded tokens and their logprobs. The
                shape of the data returned is (nactive, niter)
                where niter is the number of decoding
                iterations performed.
        """
        assert self.nactive > 0

        nactive = self.nactive
        seqlen = self.seqlen
        block_tbl = self.block_tbl
        tokens = self.tokens

        idx = bisect.bisect_left(self.graph_batch_sizes, nactive)
        assert idx < len(self.graphs)
        dg = self.graphs[idx]
        assert dg.state.batch_size >= nactive

        maxsl = int(seqlen.max().item())
        assert maxsl < self.max_seq
        if nactive == self.max_batch:
            sync_freq = self.gen_args.sync_freq
            nsteps = min(self.max_seq - maxsl, sync_freq)
        else:
            # prefer decoding many steps when the batch
            # is full; else we favor filling the batch
            # for better gpu efficiency
            nsteps = 1

        output = torch.empty(
            (nactive, nsteps),
            dtype=torch.int,
            device=self.device,
        )
        logprobs = torch.empty(
            (nactive, nsteps) if self.gen_logprobs else (0,),
            dtype=torch.float32,
            device=self.device,
        )

        dg.state.set_actual_batch_size(nactive)

        for i in range(nsteps):
            dg.state.copy_inputs(block_tbl, tokens, seqlen)
            if dg.graph:
                dg.graph.replay()
                logits: torch.Tensor = dg.logits  # type: ignore
            else:
                logits = fwd.decode(self.model, dg.state)
            logits = logits[:nactive]

            new_tokens = self._sample(self.temps[:nactive], logits)
            output[:, i] = new_tokens
            if self.gen_logprobs:
                logprobs[:, i] = -cross_entropy(
                    logits,
                    new_tokens,
                    reduction="none",
                )

            tokens = dg.state.tokens
            seqlen = dg.state.seqlen
            block_tbl = dg.state.block_tbl
            tokens[:nactive].copy_(new_tokens)
            seqlen[:nactive].add_(1)

        self.tokens[:nactive].copy_(tokens[:nactive])
        self.seqlen[:nactive].copy_(seqlen[:nactive])

        return output, logprobs.tolist()

    def _prefill(
        self,
        nprefill: int,
    ) -> tuple[torch.Tensor, list[list[float]], torch.Tensor]:
        """
        Fill the kv-caches for the added lanes and predict
        one new token per lane.

        Returns:
            ``(tokens, logprobs, toolong)`` where
            ``toolong`` is a tensor of booleans indicating
            which of the new lanes are already at their
            length limit.
        """

        new_nactive = self.nactive + nprefill

        host_cache = self.host_cache
        gpu_cache = self.cache

        cache_lanes: list[DynCacheLane] = []
        seq_info: list[tuple[int, int]] = []
        start_pos: list[int] = []
        tokens_list: list[int] = []
        seqlen_list: list[int] = []
        maxlen_list: list[int] = []
        temps_list: list[float] = []
        for i in range(self.nactive, new_nactive):
            lane = self.lane[i]
            toks = lane.all_tokens()
            blks = lane.blocks
            assert toks, "Packet.tokens must be populated"
            assert len(toks) < self.max_seq

            cache_lane = host_cache.prepare_lane(gpu_cache, blks, toks)
            cache_lanes.append(cache_lane)
            cached = host_cache.node_len * cache_lane.ready_count
            if cached == len(toks):
                cached -= 1  # sample the last token!

            seq_info.append((cached, len(toks) - cached))
            start_pos.append(len(tokens_list))
            tokens_list.extend(toks[cached:])

            self.block_tbl[i, : len(blks)] = self.tensor(blks)
            seqlen_list.append(len(toks))
            maxlen_list.append(lane.maxlen)
            temp = lane.temperature or self.gen_args.temperature
            temps_list.append(temp)

        dyn_cache = DynCache(
            gpu_cache=gpu_cache,
            host_cache=host_cache,
            cache_lanes=cache_lanes,
        )

        logger.info(f"Prefilling {len(tokens_list)} tokens")

        # split prefill in multiple calls to avoid
        # excessive memory consumption
        fst_tok, lst_tok = 0, self.max_prefill
        fst_idx = 0
        block_tbl = self.block_tbl[self.nactive : new_nactive]
        logits_list: list[torch.Tensor] = []
        tokens = self.tensor(tokens_list)
        while lst_tok < len(tokens_list):
            lst_idx = bisect.bisect_left(start_pos, lst_tok, fst_idx)
            assert 0 < lst_idx <= nprefill
            split_idx = lst_idx - 1
            cached, ntoks = seq_info[split_idx]
            new_ntoks = lst_tok - start_pos[split_idx]
            assert 0 < new_ntoks <= ntoks
            if new_ntoks == ntoks:
                new_ntoks -= 1
                lst_tok -= 1
            seq_info[split_idx] = (cached, new_ntoks)
            dyn_cache.disable("page_out")
            logits = fwd.prefill(
                model=self.model,
                token_values=tokens[fst_tok:lst_tok],
                seq_info=seq_info[fst_idx:lst_idx],
                block_tbl=block_tbl[fst_idx:lst_idx],
                block_len=self.block_len,
                cache=dyn_cache,
            )
            dyn_cache.disable("page_in")
            logits_list.append(logits[:-1])
            cached += new_ntoks
            ntoks -= new_ntoks
            seq_info[split_idx] = (cached, ntoks)
            start_pos[split_idx] += new_ntoks
            fst_idx = split_idx
            fst_tok = lst_tok
            lst_tok += self.max_prefill

            if self.profile_state != "inactive":
                # to avoid recording massive prefills
                self._profile_step()

        dyn_cache.enable("page_out")
        logits = fwd.prefill(
            model=self.model,
            token_values=tokens[fst_tok:],
            seq_info=seq_info[fst_idx:],
            block_tbl=block_tbl[fst_idx:],
            block_len=self.block_len,
            cache=dyn_cache,
        )
        logits_list.append(logits)

        logits = torch.cat(logits_list, dim=0)
        seqlen = self.tensor(seqlen_list)
        maxlen = self.tensor(maxlen_list)
        temps = torch.tensor(
            temps_list,
            dtype=torch.float,
            device=self.device,
        )

        host_cache.tick()
        assert self.cache_ready is None
        self.cache_ready = dyn_cache.host_cache_ready()

        next_tokens = self._sample(temps, logits)
        if self.gen_logprobs:
            logprobs = (
                -cross_entropy(
                    logits,
                    next_tokens,
                    reduction="none",
                )
            )[:, None].tolist()
        else:
            logprobs = []

        self.tokens[self.nactive : new_nactive] = next_tokens
        self.seqlen[self.nactive : new_nactive] = seqlen + 1
        self.maxlen[self.nactive : new_nactive] = maxlen
        self.temps[self.nactive : new_nactive] = temps
        self.nactive = new_nactive

        toolong = (seqlen + 1) >= maxlen
        return next_tokens[:, None], logprobs, toolong

    def _sample(
        self,
        temps: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.gen_args.use_sampling:
            probs = torch.softmax(logits / temps[:, None], dim=-1)
            return sampling.top_p(probs, self.gen_args.top_p)
        else:
            return torch.argmax(logits, dim=-1)

    def _kill_lane(self, lane: int, park: bool = False) -> None:
        """
        Kill a previously active generation lane. In case the
        lane killed is not the last one, the last one takes
        its place. If the ``park`` argument is True, the victim
        lane is parked.
        """
        last = self.nactive - 1
        self.nactive = last

        self.lane[lane].free(self.free_blocks)

        if park:
            self.parking.append(self.lane[lane])

        if lane == last:
            self.lane[lane] = Lane("dead")
            self.seqlen[lane] = 0
            self.maxlen[lane] = 0
            self.temps[lane] = 0
            self.block_tbl[lane] = 0
            return

        self.lane[lane] = self.lane[last]
        self.lane[last] = Lane("dead")

        self.tokens[lane] = self.tokens[last]
        self.seqlen[lane] = self.seqlen[last]
        self.seqlen[last] = 0
        self.maxlen[lane] = self.maxlen[last]
        self.maxlen[last] = 0
        self.temps[lane] = self.temps[last]
        self.temps[last] = 0
        self.block_tbl[lane] = self.block_tbl[last]
        self.block_tbl[last] = 0

    def tensor(self, seq: Iterable[int]) -> torch.Tensor:
        return torch.tensor(seq, dtype=torch.int, device=self.device)

    def _log_cuda_mem(self) -> None:
        used_gb = torch.cuda.memory_allocated() / 1e9
        if self.last_used_gb is not None:
            delta_gb = used_gb - self.last_used_gb
            logger.info(f"GPU memory: {used_gb:.3f}GB ({delta_gb:+.3f}GB)")
        else:
            logger.info(f"GPU memory: {used_gb:.3f}GB")
        self.last_used_gb = used_gb
