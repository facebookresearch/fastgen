# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


def top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): probability distribution tensor.
        p (float): probability threshold for top-p sampling.

    Returns:
        torch.Tensor: sampled indices.

    Note:
        Top-p sampling selects the smallest set whose cumulative
        probability mass exceeds the threshold p. The distribution
        is renormalized based on the selected tokens.
    """
    if p >= 1.0:
        # avoid sorting when posssible
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.squeeze(-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[probs_sum - probs_sort > p] = 0.0
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token.squeeze(-1)
