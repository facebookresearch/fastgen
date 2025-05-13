# Copyright (c) Meta Platforms, Inc. and affiliates.

import heapq
import itertools
import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import torch

from fastgen.model import Transformer as Model
from fastgen.utils.misc import hashints

logger = logging.getLogger()


class ModelCache:
    def cache_kv(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The key-value cache for layer n.
        """
        raise NotImplementedError

    def page_in(self, n: int) -> None:
        """
        Page in (host to device) the cache for layer n.
        """
        pass

    def page_out(self, n: int) -> None:
        """
        Page out (device to host) the cache for layer n.
        """
        pass


@dataclass
class RawCache(ModelCache):
    """Inference key-value caches"""

    kv_caches: list[tuple[torch.Tensor, torch.Tensor]]
    length: int
    start: int = 0

    @staticmethod
    def build(
        model: Model,
        length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "RawCache":
        """
        Allocate a cache to be used with the decoding functions.

        Args:
            model (Model): the model to build a cache for.
            length (int): per layer cache size.
                It is usually budgeted as ``max_batch * max_seq``.
            device (torch.device, optional): the device on which
                the cache should be allocated (defaults to the
                default device).
            dtype (torch.dtype, optional): the dtype to use for
                cache entries (defaults to the default dtype).
        """

        head_dim = model.args.dim // model.args.n_heads
        n_local_kv_heads = max(1, model.args.n_kv_heads // model.tp_size)
        shape = (length, n_local_kv_heads, head_dim)
        return RawCache(
            [
                (
                    torch.zeros(shape, device=device, dtype=dtype),
                    torch.zeros(shape, device=device, dtype=dtype),
                )
                for _ in range(model.args.n_layers)
            ],
            length,
        )

    @staticmethod
    def build_like(
        other: "RawCache",
        length: int,
        device: Optional[str] = None,
        pin_memory: bool = False,
    ) -> "RawCache":
        """
        Allocate a cache for the associated model of ``other``.
        The underlying tensors may be pinned to enable async
        copies.
        """
        ck = other.kv_caches[0][0]
        depth = len(other.kv_caches)
        shape = (2, depth, length) + ck.shape[1:]
        ckcv = torch.empty(
            shape,
            dtype=ck.dtype,
            device=device if device is not None else ck.device,
            pin_memory=pin_memory,
        )
        caches = [(ckcv[0, n], ckcv[1, n]) for n in range(depth)]
        return RawCache(caches, length)

    def cache_kv(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self.start, self.start + self.length
        ck = self.kv_caches[n][0][start:end]
        cv = self.kv_caches[n][1][start:end]
        return ck, cv

    def view(self, start: int, length: int) -> "RawCache":
        """
        Take a view along the sequence axis of a larger cache.

        The original cache object remains of identical size and valid
        after the shrinked alias has been used.

        Args:
            start (int): the start position in each layer's cache
            length (int): the number of items to keep in the view

        Returns:
            A view in the cache object.
        """

        assert start + length <= self.length
        view_start = self.start + start

        return RawCache(self.kv_caches, start=view_start, length=length)

    def bytes_per_token(self) -> int:
        """
        Return the number of bytes required to cache one token.
        """
        caches = self.kv_caches
        if len(caches) == 0:
            return 0

        ck = caches[0][0]
        length = ck.shape[0]
        return 2 * len(caches) * ck.numel() * ck.element_size() // length


@dataclass
class DynCacheLane:
    gpu_blocks: list[int]
    cpu_blocks: list[RawCache]
    ready_count: int

    def h2d_blocks(self) -> Iterable[tuple[int, RawCache]]:
        return itertools.islice(
            zip(self.gpu_blocks, self.cpu_blocks),
            self.ready_count,
        )

    def d2h_blocks(self) -> Iterable[tuple[int, RawCache]]:
        return itertools.islice(
            zip(self.gpu_blocks, self.cpu_blocks),
            self.ready_count,
            None,
        )


@dataclass
class DynCache(ModelCache):
    """
    A model cache class that dynamically pages
    in and out kv-caches from/to host memory.
    A ``DynCache`` object is intended to be
    used by a single ``fwd.prefill()`` call.
    """

    Mode = Literal["page_in", "page_out"]
    MODE_BITS = {
        "page_in": 1,
        "page_out": 2,
    }

    gpu_cache: RawCache
    host_cache: "Cache"
    cache_lanes: list[DynCacheLane]
    h2d_events: list[tuple[torch.cuda.Event, int]] = field(default_factory=list)
    mode: int = 3

    def disable(self, mode: Mode) -> None:
        self.mode &= ~DynCache.MODE_BITS[mode]

    def enable(self, mode: Mode) -> None:
        self.mode |= DynCache.MODE_BITS[mode]

    def page_in(self, n: int) -> None:
        if self.host_cache.disabled or not (self.mode & 1):
            return

        assert all(nn < n for _, nn in self.h2d_events)

        with torch.cuda.stream(self.host_cache.h2d_stream):
            node_len = self.host_cache.node_len
            for lane in self.cache_lanes:
                dck, dcv = self.gpu_cache.cache_kv(n)
                for gpu_block, cpu_block in lane.h2d_blocks():
                    start = gpu_block * node_len
                    end = start + node_len
                    sck, scv = cpu_block.cache_kv(n)
                    dck[start:end].copy_(sck, non_blocking=True)
                    dcv[start:end].copy_(scv, non_blocking=True)

        ev = torch.cuda.Event()
        ev.record(self.host_cache.h2d_stream)
        self.h2d_events.append((ev, n))

    def page_out(self, n: int) -> None:
        if self.host_cache.disabled or not (self.mode & 2):
            return

        ev = torch.cuda.Event()
        ev.record(torch.cuda.default_stream())

        with torch.cuda.stream(self.host_cache.d2h_stream):
            ev.wait()
            node_len = self.host_cache.node_len
            for lane in self.cache_lanes:
                sck, scv = self.gpu_cache.cache_kv(n)
                for gpu_block, cpu_block in lane.d2h_blocks():
                    start = gpu_block * node_len
                    end = start + node_len
                    dck, dcv = cpu_block.cache_kv(n)
                    dck.copy_(sck[start:end], non_blocking=True)
                    dcv.copy_(scv[start:end], non_blocking=True)

    def host_cache_ready(self) -> Optional[torch.cuda.Event]:
        """
        Return an event that must be waited on before
        the on-device caches are successfully synced
        to the host memory.
        """
        if self.host_cache.disabled:
            return None

        ev = torch.cuda.Event()
        ev.record(self.host_cache.d2h_stream)
        return ev

    def cache_kv(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.host_cache.disabled and (self.mode & 1):
            assert (
                self.h2d_events and self.h2d_events[0][1] == n
            ), f"sync event is not available for layer {n}"
            ev, _ = self.h2d_events.pop(0)
            ev.wait()  # ensure page-in is done

        return self.gpu_cache.cache_kv(n)


@dataclass
class CacheNode:
    toks_h: int
    toks: list[int]
    cache: RawCache
    clock: int
    node_id: int
    next: list["CacheNode"] = field(default_factory=list)

    def __lt__(self, other: "CacheNode") -> bool:
        # An arbitrary order to break ties in the cache
        # eviction logic. Note that we do not use Python
        # `id()` here because we need the logic to give
        # the same answer on all model-parallel ranks.
        return self.node_id < other.node_id

    def __eq__(self, other) -> bool:
        return self is other


@dataclass
class Cache:
    """Host cache for repeated prompts"""

    limit_toks: int
    node_len: int  # granularity of the trie, in tokens
    nodes: list[CacheNode] = field(default_factory=list)
    frozen: set[int] = field(default_factory=set)
    num_nodes: int = 0
    clock: int = 0
    next_id: int = 0
    downsize: float = 0.75  # downsizing factor in maintain()
    limit: int = 0  # in number of nodes
    disabled: bool = False

    free_caches: list[RawCache] = field(default_factory=list)
    h2d_stream: torch.cuda.Stream = None  # type: ignore
    d2h_stream: torch.cuda.Stream = None  # type: ignore

    def __post_init__(self):
        self.limit = self.limit_toks // self.node_len
        self.disabled = self.limit == 0
        self._create_streams()

    def _create_streams(self) -> None:
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

    def tick(self) -> None:
        "Tick the cache clock."
        self.frozen.clear()
        self.clock += 1

    def maintain(self) -> None:
        """
        Ensure the cache stays within its token limit.
        """
        if self.num_nodes <= self.limit:
            return

        target = int(self.downsize * self.limit)
        to_evict = self.num_nodes - target

        nodes: list[tuple[int, int, CacheNode, list[CacheNode]]] = []
        stk = [(n, self.nodes, 0) for n in self.nodes]
        while stk:
            n, p, d = stk.pop()
            nodes.append((n.clock, d, n, p))
            stk.extend((c, n.next, d - 1) for c in n.next)

        assert self.num_nodes == len(nodes)
        self.num_nodes -= to_evict
        logger.info(f"Cache will evict {to_evict} nodes")

        heapq.heapify(nodes)
        while to_evict > 0:
            n, p = heapq.heappop(nodes)[2:]
            assert len(n.next) == 0
            p.remove(n)
            self.free_caches.append(n.cache)
            to_evict -= 1

    def prepare_lane(
        self,
        gpu_cache: RawCache,
        blocks: list[int],
        tokens: list[int],
        no_insert: bool = False,  # for testing purposes
    ) -> DynCacheLane:
        """
        Construct a cache lane object to be used by a
        prefill call. The lane will contain references
        to cache blocks on host memory that can either
        be used to save computation or that must be
        populated during prefilling.

        The lane produced should be used to construct
        a ``DynCache`` object to pass to the prefill
        function.
        """
        if self.disabled:
            return DynCacheLane(blocks, [], 0)

        ready = 0
        caches: list[RawCache] = []
        inserting = False
        nodes = self.nodes
        while len(tokens) >= self.node_len:
            toks = tokens[: self.node_len]
            toks_h = hashints(toks)
            tokens = tokens[self.node_len :]

            if not inserting:
                for n in nodes:
                    if id(n) in self.frozen:
                        continue
                    if n.toks_h == toks_h and n.toks == toks:
                        n.clock = self.clock
                        caches.append(n.cache)
                        nodes = n.next
                        break
                else:
                    ready = len(caches)
                    inserting = True
                    if no_insert:
                        break

            if inserting:
                if self.free_caches:
                    cache = self.free_caches.pop()
                else:
                    cache = RawCache.build_like(
                        gpu_cache,
                        length=self.node_len,
                        device="cpu",
                        pin_memory=True,
                    )

                n = CacheNode(
                    toks_h=toks_h,
                    toks=toks,
                    cache=cache,
                    clock=self.clock,
                    node_id=self.next_id,
                )
                nodes.append(n)
                self.frozen.add(id(n))
                caches.append(n.cache)
                nodes = n.next
                self.num_nodes += 1
                self.next_id += 1

        ready = ready if inserting else len(caches)
        return DynCacheLane(blocks, caches, ready)

    def preallocate(self, gpu_cache: RawCache) -> None:
        """
        Preallocate the cache in host memory. Note
        that this is a surprisingly costly operation
        for large caches (e.g., 6s for 10G).
        """
        if self.disabled:
            return

        rc = RawCache.build_like(
            gpu_cache,
            length=self.node_len * self.limit,
            device="cpu",
            pin_memory=True,  # to enable async transfers
        )
        for i in range(self.limit):
            self.free_caches.append(
                rc.view(
                    start=i * self.node_len,
                    length=self.node_len,
                )
            )
