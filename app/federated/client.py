"""Federated learning client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import FLConfig


@dataclass
class ClientResult:
    """Output from a local client training pass."""

    client_id: str
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float


class FederatedClient:
    """Minimal client that trains locally and returns model weights."""

    def __init__(
        self,
        client_id: str,
        model_builder: Callable[[], nn.Module],
        config: FLConfig,
    ) -> None:
        self.client_id = client_id
        self.config = config
        self.model = model_builder()

    def load_global_weights(self, global_state: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(global_state, strict=False)

    def train_local(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
    ) -> ClientResult:
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for _ in range(self.config.local_epochs):
            for features, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(features).squeeze(-1)
                loss = criterion(outputs, targets.float())
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        num_samples = len(train_loader.dataset)

        accuracy, train_eval_loss = self._evaluate_loader(
            train_loader, criterion
        )
        val_loss = train_eval_loss
        val_accuracy = accuracy
        if validation_loader is not None:
            val_accuracy, val_loss = self._evaluate_loader(
                validation_loader, criterion
            )

        local_state = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }

        return ClientResult(
            client_id=self.client_id,
            state_dict=local_state,
            num_samples=num_samples,
            loss=avg_loss,
            accuracy=accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )

    def _evaluate_loader(
        self, loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float]:
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for features, targets in loader:
                logits = self.model(features).squeeze(-1)
                loss = criterion(logits, targets.float())
                preds = (torch.sigmoid(logits) >= 0.5).float()

                correct += int((preds == targets.float()).sum().item())
                total += int(targets.numel())
                total_loss += float(loss.item())
                batches += 1

        accuracy = float(correct / max(total, 1))
        avg_loss = float(total_loss / max(batches, 1))
        return accuracy, avg_loss
