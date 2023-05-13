"""Frequently used functions."""

from typing import Dict

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import math
import torch

Tensor: TypeAlias = torch.Tensor


def probabilities_to_dictionary(probs: Tensor) -> Dict[str, Tensor]:
    """
    Convert 1D tensor of probabilities to dictionary of probabilities of bitstrings.

    Args:
        probs (Tensor): Probabilities.

    Returns:
        Dict: Probabilities of bitstrings.

    Raises:
        ValueError: If length of probs is not a power of 2.
    """

    n = int(math.log2(len(probs)))
    if 2**n != len(probs):
        raise ValueError(f"Length of probs ({len(probs)}) is not a power of 2.")

    result = {}
    for i, p in enumerate(probs):
        bistring = bin(i)[2:].zfill(n)
        result[bistring] = p

    return result


def samples_to_dictionary(samples: Tensor) -> Dict[str, float]:
    """
    Convert 2D tensor of samples to dictionary of probabilities of bitstrings.

    Args:
        samples (Tensor): Samples.

    Returns:
        Dict: Probabilities of bitstrings.

    Raises:
        ValueError: If samples not a Tensor of integers.
    """

    if samples.is_floating_point() or samples.is_complex():
        raise ValueError("Samples must be tensors of integers.")

    bitstrings = ["".join(str(b.item()) for b in sample) for sample in samples]
    bitstring_counts = {bs: bitstrings.count(bs) for bs in set(bitstrings)}
    total = sum(bitstring_counts.values())
    bitstring_probs = {bs: count / total for bs, count in bitstring_counts.items()}

    return bitstring_probs
