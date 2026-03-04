"""Safe FedAvg aggregation utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


def _validate_inputs(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    client_sizes: Sequence[int],
) -> None:
    if not client_states:
        raise ValueError("client_states must not be empty")
    if len(client_states) != len(client_sizes):
        raise ValueError("client_states and client_sizes length mismatch")
    if any(size <= 0 for size in client_sizes):
        raise ValueError("client_sizes must be positive")


def _common_keys(
    client_states: Sequence[Mapping[str, torch.Tensor]],
) -> List[str]:
    keys = set(client_states[0].keys())
    for state in client_states[1:]:
        keys &= set(state.keys())
    return sorted(keys)


def fedavg(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    client_sizes: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """
    Aggregate model states using weighted FedAvg.

    Safety behavior:
    - Uses only keys present in all client states
    - Skips keys with shape mismatch
    - Aggregates floating tensors in float32 and casts back
    - For non-floating tensors, keeps tensor from largest client
    """

    _validate_inputs(client_states, client_sizes)

    weights = torch.tensor(client_sizes, dtype=torch.float32)
    weights = weights / weights.sum()

    aggregated: Dict[str, torch.Tensor] = {}

    for key in _common_keys(client_states):
        tensors = [state[key] for state in client_states]

        base_shape = tensors[0].shape
        if any(t.shape != base_shape for t in tensors[1:]):
            continue

        # Float tensors: weighted mean in float32 for numerical stability.
        if tensors[0].is_floating_point():
            device = tensors[0].device
            acc = torch.zeros_like(tensors[0], dtype=torch.float32, device=device)
            for w, t in zip(weights, tensors):
                acc += t.to(device=device, dtype=torch.float32) * w.to(device)
            aggregated[key] = acc.to(dtype=tensors[0].dtype)
            continue

        # Non-floating tensors (e.g. counters): use largest-client value.
        largest_idx = max(range(len(client_sizes)), key=lambda i: client_sizes[i])
        aggregated[key] = tensors[largest_idx].clone()

    if not aggregated:
        raise ValueError("No compatible parameter keys found for aggregation")

    return aggregated
