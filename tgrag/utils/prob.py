from typing import List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def sigmoid(z: float) -> float:
    """Compute the logistic sigmoid of a scalar.

    Parameters:
        z : float
            Input value.

    Returns:
        float
            1 / (1 + exp(-z)).
    """
    return 1 / (1 + np.exp(-z))


def get_importance_probability(
    pr_src: str, hc_src: str, pr_dst: str, hc_dst: str, average_importance: float
) -> float:
    """Compute a normalized importance probability for an edge.

    Parameters:
        pr_src : str
            Source node PageRank value (string representation).
        hc_src : str
            Source node hub/authority value (string representation).
        pr_dst : str
            Destination node PageRank value (string representation).
        hc_dst : str
            Destination node hub/authority value (string representation).
        average_importance : float
            Mean importance used for normalization.

    Returns:
        float
            Sigmoid-normalized importance probability.
    """
    importance = get_importance(pr_src, hc_src, pr_dst, hc_dst)
    return sigmoid(importance / average_importance)


def get_importance(pr_src: str, hc_src: str, pr_dst: str, hc_dst: str) -> float:
    """Compute the average importance of an edge (mean of |pr * hc|) from source and destination node metrics.

    Parameters:
        pr_src : str
            Source node PageRank value (string representation).
        hc_src : str
            Source node hub/authority value (string representation).
        pr_dst : str
            Destination node PageRank value (string representation).
        hc_dst : str
            Destination node hub/authority value (string representation).

    Returns:
        float
            Edge importance score.
    """
    src_importance = abs(float(pr_src)) * abs(float(hc_src))
    dst_importance = abs(float(pr_dst)) * abs(float(hc_dst))
    importance = (src_importance + dst_importance) / 2
    return importance


def get_importance_probability_node(
    pr_val: str, hc_val: str, average_importance: float
) -> float:
    """Compute a normalized importance probability for a single node.

    Parameters:
        pr_val : str
            Node PageRank value (string representation).
        hc_val : str
            Node hub/authority value (string representation).
        average_importance : float
            Mean importance used for normalization.

    Returns:
        float
            Sigmoid-normalized importance probability.
    """
    importance = get_importance_node(pr_val, hc_val)
    return sigmoid(importance / average_importance)


def get_importance_node(pr_val: str, hc_val: str) -> float:
    """Compute the importance of a single node.

    Parameters:
        pr_val : str
            Node PageRank value (string representation).
        hc_val : str
            Node hub/authority value (string representation).

    Returns:
        float
            Node importance score.
    """
    return abs(float(pr_val) * float(hc_val))


def ragged_mean_by_index(seq_list: List[torch.Tensor]) -> torch.Tensor:
    """Compute the elementwise mean across a list of variable-length 1D tensors.

    For each index i, the output contains the mean over all tensors that have
    a value at position i. Missing positions are ignored.

    Parameters:
        seq_list : List[torch.Tensor]
            List of 1D tensors, possibly of different lengths.

    Returns:
        torch.Tensor
            1D tensor where i-th entry is the mean over all seqs
             that have index i.
    """
    if not seq_list:
        return torch.empty(0)

    device = seq_list[0].device
    lengths = torch.tensor([t.numel() for t in seq_list], device=device)
    max_len = int(lengths.max().item())

    # (B, Lmax)
    padded = pad_sequence(seq_list, batch_first=True)  # expects 1D tensors
    idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, Lmax)
    mask = (idx < lengths.unsqueeze(1)).to(padded.dtype)  # (B, Lmax), 1 where valid

    sum_by_pos = (padded * mask).sum(dim=0)  # (Lmax,)
    cnt_by_pos = mask.sum(dim=0).clamp_min(1)  # avoid div/0
    return sum_by_pos / cnt_by_pos
