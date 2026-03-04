"""Federated learning server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
import torch.nn as nn

from .aggregation import fedavg
from .client import ClientResult


@dataclass
class RoundMetrics:
    """Metrics captured for each aggregation round."""

    round_number: int
    participating_clients: int
    average_client_loss: float
    average_client_accuracy: float
    average_val_loss: float
    average_val_accuracy: float


class FederatedServer:
    """Minimal server that aggregates client updates via FedAvg."""

    def __init__(self, model_builder: Callable[[], nn.Module]) -> None:
        self.global_model = model_builder()
        self.round_history: List[RoundMetrics] = []

    def global_state(self) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().clone() for k, v in self.global_model.state_dict().items()
        }

    def aggregate_round(
        self,
        round_number: int,
        client_results: List[ClientResult],
    ) -> RoundMetrics:
        if not client_results:
            raise ValueError("client_results must not be empty")

        states = [result.state_dict for result in client_results]
        sizes = [result.num_samples for result in client_results]
        aggregated = fedavg(states, sizes)

        current = self.global_model.state_dict()
        for key, tensor in aggregated.items():
            if key in current and current[key].shape == tensor.shape:
                current[key] = tensor.to(dtype=current[key].dtype)
        self.global_model.load_state_dict(current, strict=False)

        metrics = RoundMetrics(
            round_number=round_number,
            participating_clients=len(client_results),
            average_client_loss=float(
                sum(result.loss for result in client_results)
                / len(client_results)
            ),
            average_client_accuracy=float(
                sum(result.accuracy for result in client_results)
                / len(client_results)
            ),
            average_val_loss=float(
                sum(result.val_loss for result in client_results)
                / len(client_results)
            ),
            average_val_accuracy=float(
                sum(result.val_accuracy for result in client_results)
                / len(client_results)
            ),
        )
        self.round_history.append(metrics)
        return metrics
