# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import TYPE_CHECKING, Optional, TypeAlias, Union

import torch
import torch.nn

if TYPE_CHECKING:
    from src.model.transformer import Transformer as XlfModel

    from fastgen.model import Transformer as FgModel

    # the code below works for multiple model classes
    Model: TypeAlias = Union[XlfModel, FgModel]


def transplant_params(dst: torch.nn.Module, src: torch.nn.Module) -> list[str]:
    """
    Transplant model parameters from module src to module dst.

    Throws AssertionError if some parameters of the destination
    module could not be found in the source module.

    Returns:
        list[str]: state keys found in the source module that are
            not present in the destination module.
    """
    unset, extra = dst.load_state_dict(src.state_dict(), strict=False)
    assert not unset, f"missing model parameters: {unset}"
    return extra


def consecutive(*tensors: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Check whether the input tensors are consecutive in memory.

    If the input tensors are consecutive and part of a single
    storage, return a tensor that is the concatenation of all the
    input tensors on dimension 0.

    Returns:
        The concatenation of the input tensors, or None if the
        input tensors are not consecutive in memory.
    """
    if not tensors:
        return torch.Tensor()

    if not all(t.is_contiguous() for t in tensors):
        return None

    t = tensors[0]
    storage = t.untyped_storage()
    dtype = t.dtype
    sptr = storage.data_ptr()
    soff = t.storage_offset()
    ptr = t.data_ptr()
    stride = t.stride()
    d0, ds = t.shape[0], t.shape[1:]

    # from PyTorch 2.1 we can use t.nbytes() instead
    ptr += t.numel() * t.element_size()

    for t in tensors[1:]:
        if (
            t.untyped_storage().data_ptr() != sptr
            or t.dtype != dtype
            or t.data_ptr() != ptr
            or t.stride() != stride
            or t.shape[1:] != ds
        ):
            return None
        d0 += t.shape[0]
        ptr += t.numel() * t.element_size()

    cat = torch.empty(0, dtype=dtype, device=storage.device)
    return cat.set_(storage, soff, (d0, *ds), stride)


def fuse_linear_weights(model: "Model") -> None:
    """
    Make attention and feed-forward weights consecutive
    so that inference can schedule larger GEMMs to avoid
    so-called wave quantization effects.
    """

    pos = 0

    def P(w, length):
        nonlocal pos
        param = torch.nn.Parameter(w[pos : pos + length])
        pos += length
        return param

    for layer in model.layers:
        wq = layer.attention.wq.weight
        wk = layer.attention.wk.weight
        wv = layer.attention.wv.weight
        if consecutive(wq, wk, wv) is None:
            w = torch.cat((wq, wk, wv), dim=0)
            pos = 0
            layer.attention.wq.weight = P(w, wq.shape[0])
            layer.attention.wk.weight = P(w, wk.shape[0])
            layer.attention.wv.weight = P(w, wv.shape[0])

        w1 = layer.feed_forward.w1.weight
        w3 = layer.feed_forward.w3.weight
        if consecutive(w1, w3) is None:
            w = torch.cat((w1, w3), dim=0)
            pos = 0
            layer.feed_forward.w1.weight = P(w, w1.shape[0])
            layer.feed_forward.w3.weight = P(w, w3.shape[0])
